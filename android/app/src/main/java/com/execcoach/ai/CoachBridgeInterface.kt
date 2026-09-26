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
}
