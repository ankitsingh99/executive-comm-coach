package com.execcoach

import android.app.Application

class ExecCoachApplication : Application() {
    override fun onCreate() {
        super.onCreate()
        // Initialize SQLite SQLCipher native libraries
        net.sqlcipher.database.SQLiteDatabase.loadLibs(this)
    }
}
