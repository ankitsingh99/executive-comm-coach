package com.execcoach.data.local

import android.content.Context
import androidx.room.Database
import androidx.room.Room
import androidx.room.RoomDatabase
import com.execcoach.data.local.dao.CoachingDao
import com.execcoach.data.local.entity.ImprovementEntity
import com.execcoach.data.local.entity.SessionEntity
import com.execcoach.data.local.entity.StrengthEntity
import net.sqlcipher.database.SupportFactory

@Database(
    entities = [SessionEntity::class, StrengthEntity::class, ImprovementEntity::class],
    version = 1,
    exportSchema = false
)
abstract class CoachingDatabase : RoomDatabase() {

    abstract fun coachingDao(): CoachingDao

    companion object {
        @Volatile
        private var INSTANCE: CoachingDatabase? = null

        fun getInstance(context: Context, passphrase: ByteArray? = null): CoachingDatabase {
            return INSTANCE ?: synchronized(this) {
                val builder = Room.databaseBuilder(
                    context.applicationContext,
                    CoachingDatabase::class.java,
                    "exec_coach_encrypted.db"
                )

                // Attach SQLCipher hardware-backed AES-256 passphrase if provided
                if (passphrase != null) {
                    val factory = SupportFactory(passphrase)
                    builder.openHelperFactory(factory)
                }

                val instance = builder.fallbackToDestructiveMigration().build()
                INSTANCE = instance
                instance
            }
        }
    }
}
