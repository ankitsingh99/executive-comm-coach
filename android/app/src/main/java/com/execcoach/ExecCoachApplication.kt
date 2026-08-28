package com.execcoach
 
import android.app.Application
import dagger.hilt.android.HiltAndroidApp

@HiltAndroidApp
class ExecCoachApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Initialize SQLite SQLCipher native libraries
        net.sqlcipher.database.SQLiteDatabase.loadLibs(this)
    }
}
