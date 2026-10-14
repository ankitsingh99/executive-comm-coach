package com.execcoach.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class AmbientAudioServiceTest {

    @Test
    fun testSensingStateTransitions() {
        val states = SensingState.values()
        assertEquals(6, states.size)
        assertEquals(SensingState.IDLE, SensingState.valueOf("IDLE"))
        assertEquals(SensingState.DUTY_CYCLE_DSP_STANDBY, SensingState.valueOf("DUTY_CYCLE_DSP_STANDBY"))
        assertEquals(SensingState.PASSIVE_VAD_GATED, SensingState.valueOf("PASSIVE_VAD_GATED"))
        assertEquals(SensingState.CONSENT_PROMPTED, SensingState.valueOf("CONSENT_PROMPTED"))
        assertEquals(SensingState.ACTIVE_RECORDING, SensingState.valueOf("ACTIVE_RECORDING"))
        assertEquals(SensingState.POST_PROCESSING, SensingState.valueOf("POST_PROCESSING"))
    }

    @Test
    fun testServiceConstants() {
        assertEquals("exec_coach_ambient_channel", AmbientAudioService.CHANNEL_ID)
        assertEquals(1001, AmbientAudioService.NOTIFICATION_ID)
        assertEquals(1002, AmbientAudioService.HEADS_UP_NOTIFICATION_ID)
        assertEquals("ACTION_START_AMBIENT", AmbientAudioService.ACTION_START_AMBIENT)
        assertEquals("ACTION_START_ACTIVE", AmbientAudioService.ACTION_START_ACTIVE)
        assertEquals("ACTION_RELEASE_MIC_STANDBY", AmbientAudioService.ACTION_RELEASE_MIC_STANDBY)
        assertEquals("ACTION_DECLINE_NUDGE", AmbientAudioService.ACTION_DECLINE_NUDGE)
        assertEquals("ACTION_RESET_NUDGE_SUPPRESSION", AmbientAudioService.ACTION_RESET_NUDGE_SUPPRESSION)
        assertEquals("ACTION_STOP", AmbientAudioService.ACTION_STOP)
    }

    @Test
    fun testNudgeLifecycle_initialStateAllowsSpeech() {
        val lifecycle = AmbientNudgeLifecycle(silenceFramesThreshold = 10)
        org.junit.Assert.assertFalse(lifecycle.isSuppressed)
        org.junit.Assert.assertFalse(lifecycle.hasGoneSilentAfterDecline)
        // First speech frame allows nudge processing
        val allowNudge = lifecycle.onAudioFrame(isSpeechActive = true)
        org.junit.Assert.assertTrue(allowNudge)
    }

    @Test
    fun testNudgeLifecycle_declineSuppressesUntilSilenceThenRearmsOnSpeech() {
        val threshold = 10
        val lifecycle = AmbientNudgeLifecycle(silenceFramesThreshold = threshold)

        // 1. User declines nudge
        lifecycle.declineNudge()
        org.junit.Assert.assertTrue(lifecycle.isSuppressed)
        org.junit.Assert.assertFalse(lifecycle.hasGoneSilentAfterDecline)

        // 2. While conversation continues, speech frames are strictly suppressed
        for (i in 0 until 5) {
            val allowNudge = lifecycle.onAudioFrame(isSpeechActive = true)
            org.junit.Assert.assertFalse("Speech should remain suppressed while conversation continues", allowNudge)
            org.junit.Assert.assertFalse(lifecycle.hasGoneSilentAfterDecline)
        }

        // 3. Brief silence frames that don't reach threshold yet do not re-arm
        for (i in 0 until 5) {
            val allowNudge = lifecycle.onAudioFrame(isSpeechActive = false)
            org.junit.Assert.assertFalse(allowNudge)
            org.junit.Assert.assertFalse(lifecycle.hasGoneSilentAfterDecline)
        }

        // Interrupted by more speech before reaching full silence: resets silence counter
        lifecycle.onAudioFrame(isSpeechActive = true)
        org.junit.Assert.assertEquals(0, lifecycle.suppressedSilenceFrames)

        // 4. Room goes silent for a while (full threshold reached)
        for (i in 0 until threshold) {
            lifecycle.onAudioFrame(isSpeechActive = false)
        }
        org.junit.Assert.assertTrue("Should detect that conversation went silent for a while", lifecycle.hasGoneSilentAfterDecline)
        org.junit.Assert.assertTrue("Still suppressed until next speech arrives", lifecycle.isSuppressed)

        // 5. Speech is detected after this pause! Re-arms and triggers nudge
        val rearmedNudge = lifecycle.onAudioFrame(isSpeechActive = true)
        org.junit.Assert.assertTrue("Should trigger nudge when conversation is detected after brief pause", rearmedNudge)
        org.junit.Assert.assertFalse("Suppression should be cleared after re-arming", lifecycle.isSuppressed)
        org.junit.Assert.assertFalse(lifecycle.hasGoneSilentAfterDecline)
    }

    @Test
    fun testNudgeLifecycle_resetSuppression() {
        val lifecycle = AmbientNudgeLifecycle(silenceFramesThreshold = 10)
        lifecycle.declineNudge()
        org.junit.Assert.assertTrue(lifecycle.isSuppressed)

        lifecycle.resetSuppression()
        org.junit.Assert.assertFalse(lifecycle.isSuppressed)
        org.junit.Assert.assertFalse(lifecycle.hasGoneSilentAfterDecline)
        org.junit.Assert.assertEquals(0, lifecycle.suppressedSilenceFrames)
    }
}

