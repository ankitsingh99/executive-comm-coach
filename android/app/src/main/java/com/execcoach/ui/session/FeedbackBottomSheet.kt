package com.execcoach.ui.session

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.execcoach.data.local.entity.ImprovementEntity
import com.execcoach.data.local.entity.SessionEntity
import com.execcoach.data.local.entity.StrengthEntity
import com.execcoach.ui.theme.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun FeedbackBottomSheet(
    session: SessionEntity,
    strengths: List<StrengthEntity>,
    improvements: List<ImprovementEntity>,
    onDismiss: () -> Unit,
    onErase: () -> Unit
) {
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        containerColor = BackgroundDark,
        shape = RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp)
    ) {
        LazyColumn(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 20.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header: Counterpart & Strategy
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            "Executive Coaching Analysis",
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold,
                            color = TextPrimaryDark
                        )
                        Text(
                            "With ${session.counterpartName} (${session.counterpartRole})",
                            fontSize = 14.sp,
                            color = ExecutiveAccent
                        )
                    }
                    Badge(
                        containerColor = when (session.powerAxis) {
                            "UPWARD" -> ExecutiveRose
                            "LATERAL" -> ExecutiveAccent
                            else -> ExecutiveEmerald
                        }
                    ) {
                        Text(session.powerAxis, color = Color.Black, modifier = Modifier.padding(4.dp))
                    }
                }
            }

            // Score Metrics Cards
            item {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    ScoreTile("Presence", "${session.presenceScore}", ExecutiveAccent, Modifier.weight(1f))
                    ScoreTile("Assertion", "${session.assertivenessScore}", ExecutiveEmerald, Modifier.weight(1f))
                    ScoreTile("Listening", "${session.activeListeningScore}", ExecutiveAmber, Modifier.weight(1f))
                }
            }

            // Strategic Takeaway Summary
            item {
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Text("Strategic Focus", fontWeight = FontWeight.Bold, color = TextPrimaryDark)
                        Spacer(modifier = Modifier.height(4.dp))
                        Text(session.strategicSummary, fontSize = 13.sp, color = TextSecondaryDark)
                    }
                }
            }

            // Top Strengths (Top-N)
            item {
                Text(
                    "Top Strengths",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = ExecutiveEmerald
                )
            }
            items(strengths) { strength ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.CheckCircle, contentDescription = null, tint = ExecutiveEmerald, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(strength.observation, fontWeight = FontWeight.Medium, color = TextPrimaryDark, fontSize = 13.sp)
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(6.dp))
                                .background(ExecutiveNavy)
                                .padding(8.dp)
                        ) {
                            Text("\"${strength.verbatimQuote}\"", fontSize = 12.sp, fontStyle = FontStyle.Italic, color = TextSecondaryDark)
                        }
                    }
                }
            }

            // Areas for Improvement & Coached Alternatives (Top-N)
            item {
                Text(
                    "Areas for Improvement & Coached Phrasing",
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    color = ExecutiveRose
                )
            }
            items(improvements) { imp ->
                Card(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp),
                    colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
                ) {
                    Column(modifier = Modifier.padding(14.dp)) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Icon(Icons.Default.ErrorOutline, contentDescription = null, tint = ExecutiveRose, modifier = Modifier.size(16.dp))
                            Spacer(modifier = Modifier.width(6.dp))
                            Text(imp.critique, fontWeight = FontWeight.Medium, color = TextPrimaryDark, fontSize = 13.sp)
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        Text("Original Utterance:", fontSize = 11.sp, color = TextSecondaryDark)
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(6.dp))
                                .background(ExecutiveNavy)
                                .padding(8.dp)
                        ) {
                            Text("\"${imp.verbatimQuote}\"", fontSize = 12.sp, fontStyle = FontStyle.Italic, color = TextSecondaryDark)
                        }
                        Spacer(modifier = Modifier.height(6.dp))
                        Text("Executive Coached Alternative:", fontSize = 11.sp, color = ExecutiveAccent)
                        Box(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clip(RoundedCornerShape(6.dp))
                                .background(Color(0xFF0C2A44))
                                .padding(8.dp)
                        ) {
                            Text(imp.coachedPhrasing, fontSize = 12.sp, fontWeight = FontWeight.SemiBold, color = TextPrimaryDark)
                        }
                    }
                }
            }

            // DPDP Act Right to Erasure Button
            item {
                OutlinedButton(
                    onClick = {
                        onErase()
                        onDismiss()
                    },
                    modifier = Modifier.fillMaxWidth(),
                    colors = ButtonDefaults.outlinedButtonColors(contentColor = ExecutiveRose)
                ) {
                    Icon(Icons.Default.DeleteForever, contentDescription = null)
                    Spacer(modifier = Modifier.width(8.dp))
                    Text("Statutory Right to Erasure (Purge Session)")
                }
                Spacer(modifier = Modifier.height(24.dp))
            }
        }
    }
}

@Composable
fun ScoreTile(label: String, score: String, color: Color, modifier: Modifier = Modifier) {
    Card(
        modifier = modifier,
        shape = RoundedCornerShape(10.dp),
        colors = CardDefaults.cardColors(containerColor = CardBackgroundDark)
    ) {
        Column(
            modifier = Modifier.padding(10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(label, fontSize = 11.sp, color = TextSecondaryDark)
            Text(score, fontSize = 18.sp, fontWeight = FontWeight.Bold, color = color)
        }
    }
}
