package com.execcoach.service

import android.content.Context
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.MockitoJUnitRunner

@RunWith(MockitoJUnitRunner::class)
class SileroVadDetectorTest {

    @Mock
    private lateinit var mockContext: Context

    private lateinit var vadDetector: SileroVadDetector

    @Before
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        vadDetector = SileroVadDetector(mockContext)
    }

    @Test
    fun testEvaluatePcmFrame_emptyReturnsZero() {
        val prob = vadDetector.evaluatePcmFrame(ShortArray(0))
        assertEquals(0.0f, prob, 0.001f)
    }

    @Test
    fun testEvaluatePcmFrame_silenceReturnsNearZero() {
        val silentFrame = ShortArray(512) { 0 }
        val prob = vadDetector.evaluatePcmFrame(silentFrame)
        assertEquals(0.0f, prob, 0.001f)
    }

    @Test
    fun testEvaluatePcmFrame_conversationalSpeechEvaluatesAboveZero() {
        // Simulated speech signal with RMS ~ 2000
        val speechFrame = ShortArray(512) { i ->
            (2000.0 * kotlin.math.sin(2.0 * Math.PI * i / 16.0)).toInt().toShort()
        }
        val prob = vadDetector.evaluatePcmFrame(speechFrame)
        assertTrue("Speech confidence should be > 0.3 for vocal energy", prob > 0.3f)
        assertTrue("Speech confidence should be <= 1.0", prob <= 1.0f)
    }

    @Test
    fun testEvaluatePcmFrame_loudSpeechClampedToOne() {
        // High amplitude signal
        val loudFrame = ShortArray(512) { 15000 }
        val prob = vadDetector.evaluatePcmFrame(loudFrame)
        assertEquals(1.0f, prob, 0.001f)
    }

    @Test
    fun testHardwareAccelerationQueryAndClose() {
        // Verify hardware acceleration state query
        val isAccel = vadDetector.isHardwareAccelerated()
        assertFalse(isAccel) // False in JVM host mock environment

        // Verify clean disposal
        vadDetector.close()
    }

    @Test
    fun testUpdateNoiseFloor_adapts() {
        vadDetector.setNoiseFloorRms(100.0)
        assertEquals(100.0, vadDetector.getNoiseFloorRms(), 0.01)

        // Drop
        val quiet = vadDetector.updateNoiseFloor(40.0)
        assertTrue(quiet < 100.0)

        // Increase
        val noisy = vadDetector.updateNoiseFloor(150.0)
        assertTrue(noisy > quiet)
    }

    @Test
    fun testEvaluatePcmFrame_backgroundNoiseAdaptationAndRejection() {
        // Low amplitude stationary background noise (e.g. fan noise with amplitude 100)
        val noiseFrame = ShortArray(512) { 100 }
        
        // Feed background noise to establish noise floor
        repeat(10) {
            vadDetector.evaluatePcmFrame(noiseFrame)
        }

        // Stationary noise should be rejected (return 0.0)
        val noiseProb = vadDetector.evaluatePcmFrame(noiseFrame)
        assertEquals("Stationary noise should be rejected by adaptive noise floor", 0.0f, noiseProb, 0.001f)

        // Vocal speech energy well above noise floor should still be detected
        val voiceFrame = ShortArray(512) { i ->
            (2500.0 * kotlin.math.sin(2.0 * Math.PI * i / 16.0)).toInt().toShort()
        }
        val voiceProb = vadDetector.evaluatePcmFrame(voiceFrame)
        assertTrue("Speech above noise floor should be detected", voiceProb > 0.3f)
    }
}
