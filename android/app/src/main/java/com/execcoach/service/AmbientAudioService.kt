package com.execcoach.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import com.execcoach.MainActivity
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch

/**
 * Service state machine representing the lifecycle defined in PDF Pages 1-3.
 */
enum class SensingState {
    IDLE,
    DUTY_CYCLE_DSP_STANDBY, // Ultra low-power DSP Standby (Mic released, < 0.2% battery/hr)
    PASSIVE_VAD_GATED,      // Low-power ambient acoustic gating (< 0.9% battery/hr with DSP offload)
    CONSENT_PROMPTED,       // Heads-up notification / floating overlay presented to user
    ACTIVE_RECORDING,       // Explicit user-consented capture with AES-256 & Opus
    POST_PROCESSING         // ASR, Persona Binding, and LLM Coaching Evaluation
}

/**
 * State lifecycle manager for ambient speech nudging:
 * - Detects sustained speech above noise floor
 * - When user declines/dismisses a nudge: suppresses future nudges during the ongoing conversation
 * - Tracks silence duration: detects when the room has gone silent for a while
 * - When conversation resumes after the brief pause/silence, re-arms and nudges the user again.
 */
class AmbientNudgeLifecycle(
    val silenceFramesThreshold: Int = 90 // ~3.0s at 32ms frames (93 frames)
) {
    var isSuppressed: Boolean = false
        private set
    var hasGoneSilentAfterDecline: Boolean = false
        private set
    var suppressedSilenceFrames: Int = 0
        private set

    fun declineNudge() {
        isSuppressed = true
        hasGoneSilentAfterDecline = false
        suppressedSilenceFrames = 0
    }

    fun resetSuppression() {
        isSuppressed = false
        hasGoneSilentAfterDecline = false
        suppressedSilenceFrames = 0
    }

    /**
     * Evaluates a frame given whether speech is currently active.
     * Returns true if a nudge should be allowed to process/trigger.
     */
    fun onAudioFrame(isSpeechActive: Boolean): Boolean {
        if (isSpeechActive) {
            if (isSuppressed) {
                if (hasGoneSilentAfterDecline) {
                    // Conversation went silent for a while, and now speech resumes after the pause!
                    // Unsuppress and trigger nudge!
                    resetSuppression()
                    return true
                } else {
                    // Conversation ongoing from declined nudge: keep suppressed
                    suppressedSilenceFrames = 0
                    return false
                }
            } else {
                return true
            }
        } else {
            // Silence frame
            if (isSuppressed) {
                suppressedSilenceFrames++
                if (suppressedSilenceFrames >= silenceFramesThreshold) {
                    hasGoneSilentAfterDecline = true
                }
            }
            return false
        }
    }
}

/**
 * Android 14/15/16 Compliant Foreground Service with foregroundServiceType="microphone".
 * Manages deterministic transition from ambient VAD gating to active consented recording,
 * with zero mic-hogging auto-release in DSP Standby mode.
 */
class AmbientAudioService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val _currentState = MutableStateFlow(SensingState.IDLE)
    val currentState = _currentState.asStateFlow()

    private lateinit var vadDetector: SileroVadDetector
    private lateinit var audioManager: AudioRecordManager
    val nudgeLifecycle = AmbientNudgeLifecycle()

    companion object {
        const val CHANNEL_ID = "exec_coach_ambient_channel"
        const val NOTIFICATION_ID = 1001
        const val HEADS_UP_NOTIFICATION_ID = 1002
        const val ACTION_START_AMBIENT = "ACTION_START_AMBIENT"
        const val ACTION_START_ACTIVE = "ACTION_START_ACTIVE"
        const val ACTION_RELEASE_MIC_STANDBY = "ACTION_RELEASE_MIC_STANDBY"
        const val ACTION_DECLINE_NUDGE = "ACTION_DECLINE_NUDGE"
        const val ACTION_RESET_NUDGE_SUPPRESSION = "ACTION_RESET_NUDGE_SUPPRESSION"
        const val ACTION_STOP = "ACTION_STOP"

        var activeNudgeLifecycle: AmbientNudgeLifecycle? = null
    }

    override fun onCreate() {
        super.onCreate()
        activeNudgeLifecycle = nudgeLifecycle
        createNotificationChannels()
        vadDetector = SileroVadDetector(applicationContext)
        audioManager = AudioRecordManager(applicationContext)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_AMBIENT -> startPassiveAmbientGating()
            ACTION_START_ACTIVE -> startActiveConsentedRecording()
            ACTION_RELEASE_MIC_STANDBY -> releaseMicToDspStandby()
            ACTION_DECLINE_NUDGE -> declineNudge()
            ACTION_RESET_NUDGE_SUPPRESSION -> resetNudgeSuppression()
            ACTION_STOP -> stopSensingService()
        }
        return START_STICKY
    }

    fun declineNudge() {
        try {
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.cancel(HEADS_UP_NOTIFICATION_ID)
        } catch (e: Exception) {
            // Safe fallback
        }
        nudgeLifecycle.declineNudge()
        sustainedSpeechFrames = 0
        if (_currentState.value == SensingState.CONSENT_PROMPTED) {
            _currentState.value = SensingState.PASSIVE_VAD_GATED
        }
    }

    fun resetNudgeSuppression() {
        nudgeLifecycle.resetSuppression()
        sustainedSpeechFrames = 0
    }

    private var sustainedSpeechFrames = 0
    private var lastConsentNotificationMs = 0L

    private fun startPassiveAmbientGating() {
        _currentState.value = SensingState.PASSIVE_VAD_GATED
        sustainedSpeechFrames = 0

        val notification = buildOngoingPassiveNotification()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }

        serviceScope.launch {
            audioManager.startPcmStream(
                onFrameCaptured = { audioFrame16k ->
                    // Silero VAD evaluates 32ms frames with adaptive noise compensation
                    val speechProb = vadDetector.evaluatePcmFrame(audioFrame16k)
                    val isSpeech = speechProb >= 0.70f

                    if (isSpeech) {
                        val allowNudge = nudgeLifecycle.onAudioFrame(true)
                        if (allowNudge) {
                            sustainedSpeechFrames++
                            if (sustainedSpeechFrames >= 15) { // ~480ms sustained speech
                                sustainedSpeechFrames = 0
                                val now = System.currentTimeMillis()
                                if (now - lastConsentNotificationMs >= 45000L || !nudgeLifecycle.isSuppressed) {
                                    lastConsentNotificationMs = now
                                    onSustainedSpeechDetected()
                                }
                            }
                        }
                    } else {
                        sustainedSpeechFrames = (sustainedSpeechFrames - 1).coerceAtLeast(0)
                        nudgeLifecycle.onAudioFrame(false)
                    }
                },
                onSilenceTimeout = {
                    onAmbientSilenceTimeout()
                }
            )
        }
    }

    private fun onAmbientSilenceTimeout() {
        if (_currentState.value == SensingState.PASSIVE_VAD_GATED) {
            _currentState.value = SensingState.DUTY_CYCLE_DSP_STANDBY
            updatePassiveNotification("⚡ Low-Power DSP Standby (Mic Released • <0.2%/hr)")
        }
    }

    private fun releaseMicToDspStandby() {
        _currentState.value = SensingState.DUTY_CYCLE_DSP_STANDBY
        audioManager.releaseMicToStandby()
        updatePassiveNotification("⚡ Low-Power DSP Standby (Mic Disengaged)")
    }

    private fun updatePassiveNotification(contentText: String) {
        try {
            val intent = Intent(this, MainActivity::class.java)
            val pendingIntent = PendingIntent.getActivity(
                this, 0, intent,
                PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
            )
            val notification = NotificationCompat.Builder(this, CHANNEL_ID)
                .setContentTitle("Executive Coach: Low-Power DSP")
                .setContentText(contentText)
                .setSmallIcon(android.R.drawable.ic_lock_idle_charging)
                .setContentIntent(pendingIntent)
                .setPriority(NotificationCompat.PRIORITY_LOW)
                .setOngoing(true)
                .build()
            val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            notificationManager.notify(NOTIFICATION_ID, notification)
        } catch (e: Exception) {
            // Safe notification update
        }
    }

    private fun onSustainedSpeechDetected() {
        if (_currentState.value == SensingState.PASSIVE_VAD_GATED || _currentState.value == SensingState.DUTY_CYCLE_DSP_STANDBY) {
            _currentState.value = SensingState.CONSENT_PROMPTED
            showHeadsUpConsentNotification()
        }
    }

    private fun startActiveConsentedRecording() {
        _currentState.value = SensingState.ACTIVE_RECORDING
        // Play DPDP compliance audible chime to notify surrounding participants
        audioManager.playAudibleChime()
        audioManager.startEncryptedOpusSerialization()
        
        val activeNotification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Executive Coach: Active Recording")
            .setContentText("Capturing meeting dialogue with AES-256 encryption...")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setOngoing(true)
            .build()
        
        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(NOTIFICATION_ID, activeNotification)
    }

    private fun stopSensingService() {
        _currentState.value = SensingState.IDLE
        audioManager.stopCapture()
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    private fun buildOngoingPassiveNotification(): Notification {
        val intent = Intent(this, MainActivity::class.java)
        val pendingIntent = PendingIntent.getActivity(
            this, 0, intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Executive Coach: Ambient Sensing")
            .setContentText("Acoustic gate active (< 2.5% battery/hr). Waiting for dialogue...")
            .setSmallIcon(android.R.drawable.ic_lock_idle_charging)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setOngoing(true)
            .build()
    }

    private fun showHeadsUpConsentNotification() {
        val activeIntent = Intent(this, AmbientAudioService::class.java).apply {
            action = ACTION_START_ACTIVE
        }
        val pendingActive = PendingIntent.getService(
            this, 1, activeIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val declineIntent = Intent(this, AmbientAudioService::class.java).apply {
            action = ACTION_DECLINE_NUDGE
        }
        val pendingDecline = PendingIntent.getService(
            this, 2, declineIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )

        val headsUpNotification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Spoken Dialogue Detected")
            .setContentText("Tap to start executive coaching analysis for this conversation.")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setPriority(NotificationCompat.PRIORITY_DEFAULT)
            .setOnlyAlertOnce(true)
            .setSilent(true)
            .addAction(android.R.drawable.ic_media_play, "Start Coaching Session", pendingActive)
            .addAction(android.R.drawable.ic_menu_close_clear_cancel, "Decline", pendingDecline)
            .setDeleteIntent(pendingDecline)
            .setAutoCancel(true)
            .build()

        val notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        notificationManager.notify(HEADS_UP_NOTIFICATION_ID, headsUpNotification)
    }

    private fun createNotificationChannels() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "Executive Coaching Audio Service",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Foreground microphone capture and ambient acoustic gating"
            }
            val manager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            manager.createNotificationChannel(channel)
        }
    }

    override fun onDestroy() {
        if (activeNudgeLifecycle == nudgeLifecycle) {
            activeNudgeLifecycle = null
        }
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
