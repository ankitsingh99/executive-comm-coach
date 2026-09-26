package com.execcoach.service

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test

class AmbientAudioServiceTest {

    @Test
    fun testSensingStateTransitions() {
        val states = SensingState.values()
        assertEquals(5, states.size)
        assertEquals(SensingState.IDLE, SensingState.valueOf("IDLE"))
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
        assertEquals("ACTION_STOP", AmbientAudioService.ACTION_STOP)
    }
}
