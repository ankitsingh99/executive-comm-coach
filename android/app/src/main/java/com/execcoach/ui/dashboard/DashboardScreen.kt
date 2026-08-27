package com.execcoach.ui.dashboard

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.execcoach.data.local.entity.SessionEntity
import com.execcoach.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DashboardScreen(
    sessions: List<SessionEntity>,
    isAmbientActive: Boolean,
    onToggleAmbient: () -> Unit,
    onSessionSelected: (SessionEntity) -> Unit,
    onEraseSession: (String) -> Unit
) {
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Column {
                        Text(
                            "Executive Presence",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onBackground
                        )
                        Text(
                            "On-Device AI Coaching Engine",
                            fontSize = 12.sp,
                            color = TextSecondaryDark
                        )
                    }
                },
                actions = {
                    IconButton(onClick = { /* Settings / DPDP Audit */ }) {
                        Icon(Icons.Default.Security, contentDescription = "DPDP Compliance", tint = ExecutiveEmerald)
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background)
            )
        },
        containerColor = MaterialTheme.colorScheme.background
    ) { padding ->
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Ambient Sensing Control Banner
            item {
                AmbientStatusCard(
                    isAmbientActive = isAmbientActive,
                    onToggle = onToggleAmbient
                )
            }

            // High-level Quantitative KPI Metrics
            item {
                Text(
                    "Longitudinal Performance",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onBackground
                )
                Spacer(modifier = Modifier.height(8.dp))
                MetricsOverviewRow(sessions)
            }

            // Recent Coaching Sessions
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        "Recent Conversations",
                        fontSize = 16.sp,
                        fontWeight = FontWeight.SemiBold,
                        color = MaterialTheme.colorScheme.onBackground
                    )
                    Text(
                        "${sessions.size} recorded",
                        fontSize = 12.sp,
                        color = TextSecondaryDark
                    )
                }
            }

            if (sessions.isEmpty()) {
                item {
                    EmptySessionsCard()
                }
            } else {
                items(sessions, key = { it.sessionId }) { session ->
                    SessionItemCard(
                        session = session,
                        onClick = { onSessionSelected(session) },
                        onErase = { onEraseSession(session.sessionId) }
                    )
                }
            }
            item { Spacer(modifier = Modifier.height(24.dp)) }
        }
    }
}

@Composable
fun AmbientStatusCard(
    isAmbientActive: Boolean,
    onToggle: () -> Unit
) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Box(
                    modifier = Modifier
                        .size(12.dp)
                        .clip(CircleShape)
                        .background(if (isAmbientActive) ExecutiveEmerald else ExecutiveRose)
                )
                Spacer(modifier = Modifier.width(12.dp))
                Column {
                    Text(
                        if (isAmbientActive) "Ambient Sensing Active" else "Ambient Sensing Paused",
                        fontWeight = FontWeight.SemiBold,
                        color = TextPrimaryDark
                    )
                    Text(
                        if (isAmbientActive) "Silero VAD Gating (< 2.5% battery/hr)" else "Tap start to listen for meetings",
                        fontSize = 12.sp,
                        color = TextSecondaryDark
                    )
                }
            }
            Switch(
                checked = isAmbientActive,
                onCheckedChange = { onToggle() },
                colors = SwitchDefaults.colors(
                    checkedThumbColor = ExecutiveEmerald,
                    checkedTrackColor = ExecutiveNavy
                )
            )
        }
    }
}

@Composable
fun MetricsOverviewRow(sessions: List<SessionEntity>) {
    val avgPresence = if (sessions.isNotEmpty()) sessions.map { it.presenceScore }.average().toInt() else 78
    val avgAssertiveness = if (sessions.isNotEmpty()) sessions.map { it.assertivenessScore }.average().toInt() else 82
    val avgListening = if (sessions.isNotEmpty()) sessions.map { it.activeListeningScore }.average().toInt() else 74

    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.spacedBy(8.dp)
    ) {
        MetricScoreBox("Presence", "$avgPresence", ExecutiveAccent, Modifier.weight(1f))
        MetricScoreBox("Assertion", "$avgAssertiveness", ExecutiveEmerald, Modifier.weight(1f))
        MetricScoreBox("Listening", "$avgListening", ExecutiveAmber, Modifier.weight(1f))
    }
}

@Composable
fun MetricScoreBox(title: String, score: String, color: Color, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
    ) {
        Column(
            modifier = Modifier.padding(12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(title, fontSize = 12.sp, color = TextSecondaryDark)
            Spacer(modifier = Modifier.height(4.dp))
            Text(score, fontSize = 22.sp, fontWeight = FontWeight.Bold, color = color)
            Text("/100", fontSize = 10.sp, color = TextSecondaryDark)
        }
    }
}

@Composable
fun SessionItemCard(
    session: SessionEntity,
    onClick: () -> Unit,
    onErase: () -> Unit
) {
    Card(
        onClick = onClick,
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(14.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
    ) {
        Column(modifier = Modifier.padding(16.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        session.counterpartName,
                        fontWeight = FontWeight.Bold,
                        fontSize = 16.sp,
                        color = TextPrimaryDark
                    )
                    Text(
                        "${session.counterpartRole} • ${session.powerAxis}",
                        fontSize = 12.sp,
                        color = ExecutiveAccent
                    )
                }
                IconButton(onClick = onErase) {
                    Icon(
                        Icons.Default.DeleteOutline,
                        contentDescription = "DPDP Erasure",
                        tint = ExecutiveRose
                    )
                }
            }
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                session.strategicSummary,
                fontSize = 13.sp,
                color = TextSecondaryDark,
                maxLines = 2
            )
            Spacer(modifier = Modifier.height(10.dp))
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Text("Presence: ${session.presenceScore}/100", fontSize = 12.sp, color = ExecutiveEmerald)
                Text("Fillers: ${session.fillerCountTotal}", fontSize = 12.sp, color = ExecutiveAmber)
            }
        }
    }
}

@Composable
fun EmptySessionsCard() {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(12.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
    ) {
        Column(
            modifier = Modifier
                .padding(24.dp)
                .fillMaxWidth(),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Icon(Icons.Default.MicNone, contentDescription = null, tint = TextSecondaryDark, modifier = Modifier.size(40.dp))
            Spacer(modifier = Modifier.height(8.dp))
            Text("No Coaching Sessions Yet", fontWeight = FontWeight.SemiBold, color = TextPrimaryDark)
            Text(
                "Ambient microphone sensing will trigger an instant coaching card when your next conversation finishes.",
                fontSize = 12.sp,
                color = TextSecondaryDark,
                textAlign = androidx.compose.ui.text.style.TextAlign.Center
            )
        }
    }
}
