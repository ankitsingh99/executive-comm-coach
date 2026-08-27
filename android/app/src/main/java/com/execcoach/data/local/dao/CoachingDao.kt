package com.execcoach.data.local.dao

import androidx.room.*
import com.execcoach.data.local.entity.ImprovementEntity
import com.execcoach.data.local.entity.SessionEntity
import com.execcoach.data.local.entity.StrengthEntity
import kotlinx.coroutines.flow.Flow

@Dao
interface CoachingDao {

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertSession(session: SessionEntity)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertStrengths(strengths: List<StrengthEntity>)

    @Insert(onConflict = OnConflictStrategy.REPLACE)
    suspend fun insertImprovements(improvements: List<ImprovementEntity>)

    @Query("SELECT * FROM coaching_sessions ORDER BY timestampIso DESC")
    fun getAllSessionsFlow(): Flow<List<SessionEntity>>

    @Query("SELECT * FROM coaching_sessions WHERE sessionId = :sessionId")
    suspend fun getSessionById(sessionId: String): SessionEntity?

    @Query("SELECT * FROM coaching_strengths WHERE sessionId = :sessionId")
    suspend fun getStrengthsForSession(sessionId: String): List<StrengthEntity>

    @Query("SELECT * FROM coaching_improvements WHERE sessionId = :sessionId")
    suspend fun getImprovementsForSession(sessionId: String): List<ImprovementEntity>

    // DPDP Act 2023 Statutory Right to Erasure
    @Query("DELETE FROM coaching_sessions WHERE sessionId = :sessionId")
    suspend fun deleteSession(sessionId: String)

    @Query("DELETE FROM coaching_strengths WHERE sessionId = :sessionId")
    suspend fun deleteStrengths(sessionId: String)

    @Query("DELETE FROM coaching_improvements WHERE sessionId = :sessionId")
    suspend fun deleteImprovements(sessionId: String)

    @Transaction
    suspend fun eraseSessionCompletely(sessionId: String) {
        deleteSession(sessionId)
        deleteStrengths(sessionId)
        deleteImprovements(sessionId)
    }
}
