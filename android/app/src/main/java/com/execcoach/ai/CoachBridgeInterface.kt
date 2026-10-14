package com.execcoach.ai

import android.content.Context
import android.webkit.JavascriptInterface

/**
 * JavaScript Interface bridge to expose 100% On-Device AI capabilities to the WebView frontend.
 */
class CoachBridgeInterface(private val context: Context, private val engine: OnDeviceCoachEngine) {

    @JavascriptInterface
    fun isAvailable(): Boolean {
        return true
    }

    @JavascriptInterface
    fun evaluateConversation(dialogueText: String): String {
        return try {
            engine.evaluateConversation(dialogueText)
        } catch (e: Exception) {
            """{"status":"error","message":"${e.localizedMessage}"}"""
        }
    }

    @JavascriptInterface
    fun getVoiceprints(): String {
        return try {
            engine.getSavedVoiceprints()
        } catch (e: Exception) {
            "[]"
        }
    }

    @JavascriptInterface
    fun saveVoiceprint(jsonString: String): Boolean {
        return try {
            engine.saveVoiceprint(jsonString)
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun eraseVoiceprint(speakerName: String): Boolean {
        return try {
            engine.eraseVoiceprint(speakerName)
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun setAppTheme(themeMode: String): Boolean {
        return try {
            if (context is com.execcoach.MainActivity) {
                val isDark = when (themeMode) {
                    "dark" -> true
                    "light" -> false
                    else -> {
                        val nightModeFlags = context.resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK
                        nightModeFlags == android.content.res.Configuration.UI_MODE_NIGHT_YES
                    }
                }
                context.updateSystemBarTheme(isDark)
            }
            true
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun getSystemTheme(): String {
        return try {
            val nightModeFlags = context.resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK
            if (nightModeFlags == android.content.res.Configuration.UI_MODE_NIGHT_YES) "dark" else "light"
        } catch (e: Exception) {
            "dark"
        }
    }

    @JavascriptInterface
    fun hasMicPermission(): Boolean {
        return try {
            androidx.core.content.ContextCompat.checkSelfPermission(
                context,
                android.Manifest.permission.RECORD_AUDIO
            ) == android.content.pm.PackageManager.PERMISSION_GRANTED
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun requestMicPermission(): Boolean {
        return try {
            if (context is com.execcoach.MainActivity) {
                context.runOnUiThread {
                    context.requestAudioPermissions()
                }
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun getHardwareDspInfo(): String {
        return try {
            val audioManager = com.execcoach.service.AudioRecordManager(context)
            audioManager.getHardwareDspInfoJson()
        } catch (e: Exception) {
            """{"hardwareDspSupported":false,"dspArchitecture":"Software VAD","currentState":"IDLE","isMicHogged":false,"isMicReleased":true,"silenceThresholdSec":8,"estimatedPowerDrain":"< 0.2% / hr (Zero Mic Hogging)","aecAvailable":false,"nsAvailable":false,"agcAvailable":false}"""
        }
    }

    @JavascriptInterface
    fun isMicHogged(): Boolean {
        return try {
            val audioManager = com.execcoach.service.AudioRecordManager(context)
            audioManager.isMicHogged()
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun startNativeSpeechRecognition(): Boolean {
        return try {
            if (context is com.execcoach.MainActivity) {
                context.startSpeechRecognition { text, isFinal ->
                    context.sendSpeechToWebView(text, isFinal)
                }
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun stopNativeSpeechRecognition(): Boolean {
        return try {
            if (context is com.execcoach.MainActivity) {
                context.stopSpeechRecognition()
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun startAmbientService(): Boolean {
        return try {
            if (context is com.execcoach.MainActivity) {
                context.startAmbientServiceIfPermitted()
                true
            } else {
                false
            }
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun stopAmbientService(): Boolean {
        return try {
            val serviceIntent = android.content.Intent(context, com.execcoach.service.AmbientAudioService::class.java).apply {
                action = com.execcoach.service.AmbientAudioService.ACTION_STOP
            }
            context.startService(serviceIntent)
            true
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun declineNudge(): Boolean {
        return try {
            val serviceIntent = android.content.Intent(context, com.execcoach.service.AmbientAudioService::class.java).apply {
                action = com.execcoach.service.AmbientAudioService.ACTION_DECLINE_NUDGE
            }
            context.startService(serviceIntent)
            com.execcoach.service.AmbientAudioService.activeNudgeLifecycle?.declineNudge()
            true
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun isNudgeSuppressed(): Boolean {
        return com.execcoach.service.AmbientAudioService.activeNudgeLifecycle?.isSuppressed ?: false
    }

    @JavascriptInterface
    fun resetNudgeSuppression(): Boolean {
        return try {
            val serviceIntent = android.content.Intent(context, com.execcoach.service.AmbientAudioService::class.java).apply {
                action = com.execcoach.service.AmbientAudioService.ACTION_RESET_NUDGE_SUPPRESSION
            }
            context.startService(serviceIntent)
            com.execcoach.service.AmbientAudioService.activeNudgeLifecycle?.resetSuppression()
            true
        } catch (e: Exception) {
            false
        }
    }

    @JavascriptInterface
    fun getAdaptiveNoiseFloor(): Double {
        return try {
            val audioManager = com.execcoach.service.AudioRecordManager(context)
            audioManager.getAdaptiveNoiseFloorRms()
        } catch (e: Exception) {
            120.0
        }
    }

    @JavascriptInterface
    fun getAdaptiveSilenceThreshold(): Double {
        return try {
            val audioManager = com.execcoach.service.AudioRecordManager(context)
            audioManager.getAdaptiveSilenceThresholdRms()
        } catch (e: Exception) {
            350.0
        }
    }
}


