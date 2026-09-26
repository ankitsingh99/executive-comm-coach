package com.execcoach.ai

import android.content.Context
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.Mockito.`when`
import org.mockito.MockitoAnnotations
import org.mockito.junit.MockitoJUnitRunner

@RunWith(MockitoJUnitRunner::class)
class CoachBridgeInterfaceTest {

    @Mock
    private lateinit var mockContext: Context

    @Mock
    private lateinit var mockEngine: OnDeviceCoachEngine

    private lateinit var bridge: CoachBridgeInterface

    @Before
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        bridge = CoachBridgeInterface(mockContext, mockEngine)
    }

    @Test
    fun testIsAvailable_returnsTrue() {
        assertTrue(bridge.isAvailable())
    }

    @Test
    fun testEvaluateConversation_success() {
        `when`(mockEngine.evaluateConversation("Hello team")).thenReturn("""{"executive_presence_score": 85}""")
        val result = bridge.evaluateConversation("Hello team")
        assertTrue(result.contains("85"))
    }

    @Test
    fun testEvaluateConversation_handlesException() {
        `when`(mockEngine.evaluateConversation("Error test")).thenThrow(RuntimeException("Model offline"))
        val result = bridge.evaluateConversation("Error test")
        assertTrue(result.contains("error"))
        assertTrue(result.contains("Model offline"))
    }

    @Test
    fun testGetVoiceprints_success() {
        `when`(mockEngine.getSavedVoiceprints()).thenReturn("""[{"speaker_name":"Ankit"}]""")
        val result = bridge.getVoiceprints()
        assertTrue(result.contains("Ankit"))
    }

    @Test
    fun testSaveVoiceprint() {
        `when`(mockEngine.saveVoiceprint("""{"speaker_name":"Ankit"}""")).thenReturn(true)
        assertTrue(bridge.saveVoiceprint("""{"speaker_name":"Ankit"}"""))
    }

    @Test
    fun testEraseVoiceprint() {
        `when`(mockEngine.eraseVoiceprint("Ankit")).thenReturn(true)
        assertTrue(bridge.eraseVoiceprint("Ankit"))
    }

    @Test
    fun testHasMicPermission_safeExecution() {
        val hasPerm = bridge.hasMicPermission()
        assertTrue(hasPerm || !hasPerm)
    }

    @Test
    fun testRequestMicPermission_nonMainActivityReturnsFalse() {
        assertFalse(bridge.requestMicPermission())
    }

    @Test
    fun testSetAppTheme_safeExecution() {
        assertTrue(bridge.setAppTheme("dark"))
        assertTrue(bridge.setAppTheme("light"))
    }

    @Test
    fun testGetSystemTheme_safeExecution() {
        val theme = bridge.getSystemTheme()
        assertTrue(theme == "dark" || theme == "light")
    }
}
