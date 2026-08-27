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
    PASSIVE_VAD_GATED,      // Low-power ambient acoustic gating (< 2.5% battery/hr)
    CONSENT_PROMPTED,       // Heads-up notification / floating overlay presented to user
    ACTIVE_RECORDING,       // Explicit user-consented capture with AES-256 & Opus
    POST_PROCESSING         // ASR, Persona Binding, and LLM Coaching Evaluation
}

/**
 * Android 14/15/16 Compliant Foreground Service with foregroundServiceType="microphone".
 * Manages deterministic transition from ambient VAD gating to active consented recording.
 */
class AmbientAudioService : Service() {

    private val serviceScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val _currentState = MutableStateFlow(SensingState.IDLE)
    val currentState = _currentState.asStateFlow()

    private lateinit var vadDetector: SileroVadDetector
    private lateinit var audioManager: AudioRecordManager

    companion object {
        const val CHANNEL_ID = "exec_coach_ambient_channel"
        const val NOTIFICATION_ID = 1001
        const val HEADS_UP_NOTIFICATION_ID = 1002
        const val ACTION_START_AMBIENT = "ACTION_START_AMBIENT"
        const val ACTION_START_ACTIVE = "ACTION_START_ACTIVE"
        const val ACTION_STOP = "ACTION_STOP"
    }

    override fun onCreate() {
        super.onCreate()
        createNotificationChannels()
        vadDetector = SileroVadDetector(applicationContext)
        audioManager = AudioRecordManager(applicationContext)
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_START_AMBIENT -> startPassiveAmbientGating()
            ACTION_START_ACTIVE -> startActiveConsentedRecording()
            ACTION_STOP -> stopSensingService()
        }
        return START_STICKY
    }

    private fun startPassiveAmbientGating() {
        _currentState.value = SensingState.PASSIVE_VAD_GATED

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
            audioManager.startPcmStream { audioFrame16k ->
                // Silero VAD evaluates 32ms frames in <1ms
                val speechProb = vadDetector.evaluatePcmFrame(audioFrame16k)
                if (speechProb >= 0.75f) {
                    onSustainedSpeechDetected()
                }
            }
        }
    }

    private fun onSustainedSpeechDetected() {
        if (_currentState.value == SensingState.PASSIVE_VAD_GATED) {
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

        val headsUpNotification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Spoken Dialogue Detected")
            .setContentText("Tap to start executive coaching analysis for this conversation.")
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setDefaults(Notification.DEFAULT_ALL)
            .addAction(android.R.drawable.ic_media_play, "Start Coaching Session", pendingActive)
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
        serviceScope.cancel()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
