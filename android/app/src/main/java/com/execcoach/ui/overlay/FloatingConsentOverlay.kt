package com.execcoach.ui.overlay

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Mic
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Stop
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.execcoach.ui.theme.ExecutiveAccent
import com.execcoach.ui.theme.ExecutiveEmerald
import com.execcoach.ui.theme.ExecutiveNavy
import com.execcoach.ui.theme.ExecutiveRose

/**
 * Floating Action Overlay Bubble rendered via SYSTEM_ALERT_WINDOW permission
 * to provide non-intrusive ambient speech detection nudges & consented recording toggle.
 */
@Composable
fun FloatingConsentOverlay(
    isRecording: Boolean,
    dialogueDetected: Boolean = true,
    onStartConsentedSession: () -> Unit,
    onStopSession: () -> Unit,
    onDismissNudge: () -> Unit = {}
) {
    Surface(
        modifier = Modifier.wrapContentSize(),
        shape = RoundedCornerShape(20.dp),
        color = ExecutiveNavy,
        tonalElevation = 10.dp,
        shadowElevation = 8.dp
    ) {
        Column(
            modifier = Modifier.padding(horizontal = 14.dp, vertical = 10.dp)
        ) {
            Row(
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
                Column {
                    Text(
                        text = if (isRecording) "Recording • DPDP Encrypted" else "🎙️ Spoken Dialogue Detected",
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        color = Color.White
                    )
                    Text(
                        text = if (isRecording) "Capturing for executive analysis" else "Start recording for coach analysis?",
                        fontSize = 10.sp,
                        color = Color.LightGray
                    )
                }
            }

            if (!isRecording && dialogueDetected) {
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.End,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    TextButton(
                        onClick = onDismissNudge,
                        contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp)
                    ) {
                        Text("Dismiss", color = Color.Gray, fontSize = 11.sp)
                    }
                    Spacer(modifier = Modifier.width(6.dp))
                    Button(
                        onClick = onStartConsentedSession,
                        colors = ButtonDefaults.buttonColors(containerColor = ExecutiveEmerald),
                        shape = RoundedCornerShape(12.dp),
                        contentPadding = PaddingValues(horizontal = 10.dp, vertical = 4.dp)
                    ) {
                        Icon(Icons.Default.PlayArrow, contentDescription = null, modifier = Modifier.size(14.dp), tint = Color.White)
                        Spacer(modifier = Modifier.width(4.dp))
                        Text("Start Analysis", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                    }
                }
            }
        }
    }
}

