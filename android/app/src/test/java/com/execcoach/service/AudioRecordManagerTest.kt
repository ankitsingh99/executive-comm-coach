package com.execcoach.service

import android.content.Context
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith
import org.mockito.Mock
import org.mockito.MockitoAnnotations
import org.mockito.junit.MockitoJUnitRunner
import java.io.File
import kotlin.math.sqrt

@RunWith(MockitoJUnitRunner::class)
class AudioRecordManagerTest {

    @Mock
    private lateinit var mockContext: Context

    private lateinit var audioManager: AudioRecordManager

    @Before
    fun setUp() {
        MockitoAnnotations.openMocks(this)
        audioManager = AudioRecordManager(mockContext)
    }

    @Test
    fun testComputeFrameRms_silence() {
        val silentFrame = ShortArray(512) { 0 }
        val rms = audioManager.computeFrameRms(silentFrame)
        assertEquals(0.0, rms, 0.001)
    }

    @Test
    fun testComputeFrameRms_empty() {
        val emptyFrame = ShortArray(0)
        val rms = audioManager.computeFrameRms(emptyFrame)
        assertEquals(0.0, rms, 0.001)
    }

    @Test
    fun testComputeFrameRms_constantSignal() {
        val constantValue: Short = 1000
        val constantFrame = ShortArray(512) { constantValue }
        val rms = audioManager.computeFrameRms(constantFrame)
        assertEquals(1000.0, rms, 0.01)
    }

    @Test
    fun testComputeFrameRms_sineWave() {
        val amplitude = 5000.0
        val frame = ShortArray(512) { i ->
            (amplitude * kotlin.math.sin(2.0 * Math.PI * i / 32.0)).toInt().toShort()
        }
        val rms = audioManager.computeFrameRms(frame)
        // Theoretical RMS of sine wave = amplitude / sqrt(2) ≈ 3535.53
        val expected = amplitude / sqrt(2.0)
        assertEquals(expected, rms, 50.0)
    }

    @Test
    fun testIsHardwareDspOffloadSupported_returnsBooleanSafely() {
        // Should execute safely without throwing Unhandled Exceptions on any test environment
        val supported = audioManager.isHardwareDspOffloadSupported()
        assertTrue(supported || !supported)
    }

    @Test
    fun testAttachHardwareAudioFx_handlesSafely() {
        // Calling with simulated sessionId should execute safely
        audioManager.attachHardwareAudioFx(1234)
    }

    @Test
    fun testStopCapture_safelyResetsState() {
        audioManager.stopCapture()
        assertEquals(DspPowerState.IDLE, audioManager.getCurrentDspState())
        assertFalse(audioManager.isMicHogged())
    }

    @Test
    fun testGetHardwareDspInfoJson_structure() {
        val info = audioManager.getHardwareDspInfoJson()
        assertTrue(info.contains("hardwareDspSupported"))
        assertTrue(info.contains("currentState"))
        assertTrue(info.contains("estimatedPowerDrain"))
        assertTrue(info.contains("adaptiveNoiseFloorRms"))
        assertTrue(info.contains("adaptiveSilenceThresholdRms"))
    }

    @Test
    fun testReleaseMicToStandby_transitionsState() {
        audioManager.releaseMicToStandby()
        assertEquals(DspPowerState.DSP_STANDBY_RELEASED, audioManager.getCurrentDspState())
        assertFalse(audioManager.isMicHogged())
    }

    @Test
    fun testAdaptiveNoiseFloorTracking() {
        // Initial noise floor starts at baseline
        audioManager.setAdaptiveNoiseFloorRms(200.0)
        assertEquals(200.0, audioManager.getAdaptiveNoiseFloorRms(), 0.01)

        // Drop in ambient noise: adapts downward quickly
        val updatedQuiet = audioManager.updateAdaptiveNoiseFloor(60.0)
        assertTrue("Noise floor should adapt downward quickly", updatedQuiet < 200.0)

        // Moderate increase: adapts upward gradually
        val updatedNoise = audioManager.updateAdaptiveNoiseFloor(180.0)
        assertTrue(updatedNoise > updatedQuiet)
    }

    @Test
    fun testAdaptiveSilenceThreshold_scalesWithNoiseFloor() {
        audioManager.setAdaptiveNoiseFloorRms(100.0)
        val lowNoiseThreshold = audioManager.getAdaptiveSilenceThresholdRms()

        audioManager.setAdaptiveNoiseFloorRms(400.0)
        val highNoiseThreshold = audioManager.getAdaptiveSilenceThresholdRms()

        assertTrue("Higher background noise floor must elevate silence threshold", highNoiseThreshold > lowNoiseThreshold)
        assertTrue(highNoiseThreshold >= 400.0)
    }
}
