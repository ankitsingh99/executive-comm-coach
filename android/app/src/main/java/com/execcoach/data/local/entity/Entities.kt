package com.execcoach.data.local.entity

import androidx.room.Entity
import androidx.room.PrimaryKey

@Entity(tableName = "coaching_sessions")
data class SessionEntity(
    @PrimaryKey val sessionId: String,
    val timestampIso: String,
    val counterpartName: String,
    val counterpartRole: String,
    val powerAxis: String, // UPWARD, LATERAL, DOWNWARD
    val presenceScore: Int,
    val assertivenessScore: Int,
    val activeListeningScore: Int,
    val fillerCountTotal: Int,
    val strategicSummary: String,
    val personaAlignmentNotes: String,
    val encryptedAudioPath: String?
)

@Entity(tableName = "coaching_strengths")
data class StrengthEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val observation: String,
    val verbatimQuote: String
)

@Entity(tableName = "coaching_improvements")
data class ImprovementEntity(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val sessionId: String,
    val critique: String,
    val verbatimQuote: String,
    val coachedPhrasing: String
)
