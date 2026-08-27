package com.execcoach.ui.overlay

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.execcoach.ui.theme.ExecutiveAccent
import com.execcoach.ui.theme.ExecutiveEmerald
import com.execcoach.ui.theme.ExecutiveNavy
import com.execcoach.ui.theme.ExecutiveRose

/**
 * Floating Action Overlay Bubble rendered via SYSTEM_ALERT_WINDOW permission
 * to provide non-intrusive ambient status & active recording toggle.
 */
@Composable
fun FloatingConsentOverlay(
    isRecording: Boolean,
    onStartConsentedSession: () -> Unit,
    onStopSession: () -> Unit
) {
    Surface(
        modifier = Modifier.wrapContentSize(),
        shape = RoundedCornerShape(24.dp),
        color = ExecutiveNavy,
        tonalElevation = 8.dp
    ) {
        Row(
            modifier = Modifier
                .padding(horizontal = 14.dp, vertical = 8.dp),
            verticalAlignment = Alignment.CenterVertically
        ) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(CircleShape)
                    .background(if (isRecording) ExecutiveRose else ExecutiveEmerald)
                    .clickable {
                        if (isRecording) onStopSession() else onStartConsentedSession()
                    },
                contentAlignment = Alignment.Center
            ) {
                Icon(
                    imageVector = if (isRecording) Icons.Default.Stop else Icons.Default.Mic,
                    contentDescription = "Consent Overlay",
                    tint = Color.White,
                    modifier = Modifier.size(20.dp)
                )
            }
            Spacer(modifier = Modifier.width(10.dp))
            Text(
                text = if (isRecording) "Recording • DPDP Encrypted" else "Dialogue Detected • Tap to Coach",
                fontSize = 12.sp,
                color = Color.White
            )
        }
    }
}
