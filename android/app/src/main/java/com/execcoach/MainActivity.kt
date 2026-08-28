package com.execcoach

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.*
import androidx.core.content.ContextCompat
import com.execcoach.data.local.CoachingDatabase
import com.execcoach.data.local.entity.ImprovementEntity
import com.execcoach.data.local.entity.SessionEntity
import com.execcoach.data.local.entity.StrengthEntity
import com.execcoach.service.AmbientAudioService
import com.execcoach.ui.dashboard.DashboardScreen
import com.execcoach.ui.session.FeedbackBottomSheet
import com.execcoach.ui.theme.ExecCoachTheme
import dagger.hilt.android.AndroidEntryPoint
import kotlinx.coroutines.launch

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private val database by lazy { CoachingDatabase.getInstance(applicationContext) }
    private var isAmbientServiceRunning by mutableStateOf(false)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val recordGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
        if (recordGranted) {
            toggleAmbientService()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            ExecCoachTheme {
                val coroutineScope = rememberCoroutineScope()
                val sessions by database.coachingDao().getAllSessionsFlow().collectAsState(initial = emptyList())
                var selectedSession by remember { mutableStateOf<SessionEntity?>(null) }
                var selectedStrengths by remember { mutableStateOf<List<StrengthEntity>>(emptyList()) }
                var selectedImprovements by remember { mutableStateOf<List<ImprovementEntity>>(emptyList()) }

                DashboardScreen(
                    sessions = sessions,
                    isAmbientActive = isAmbientServiceRunning,
                    onToggleAmbient = { requestPermissionsAndToggle() },
                    onSessionSelected = { session ->
                        coroutineScope.launch {
                            selectedStrengths = database.coachingDao().getStrengthsForSession(session.sessionId)
                            selectedImprovements = database.coachingDao().getImprovementsForSession(session.sessionId)
                            selectedSession = session
                        }
                    },
                    onEraseSession = { sessionId ->
                        coroutineScope.launch {
                            database.coachingDao().eraseSessionCompletely(sessionId)
                        }
                    }
                )

                selectedSession?.let { session ->
                    FeedbackBottomSheet(
                        session = session,
                        strengths = selectedStrengths,
                        improvements = selectedImprovements,
                        onDismiss = { selectedSession = null },
                        onErase = {
                            coroutineScope.launch {
                                database.coachingDao().eraseSessionCompletely(session.sessionId)
                            }
                        }
                    )
                }
            }
        }
    }

    private fun requestPermissionsAndToggle() {
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val allGranted = permissions.all {
            ContextCompat.checkSelfPermission(this, it) == PackageManager.PERMISSION_GRANTED
        }
        if (allGranted) {
            toggleAmbientService()
        } else {
            permissionLauncher.launch(permissions.toTypedArray())
        }
    }

    private fun toggleAmbientService() {
        val intent = Intent(this, AmbientAudioService::class.java)
        if (isAmbientServiceRunning) {
            intent.action = AmbientAudioService.ACTION_STOP
            startService(intent)
            isAmbientServiceRunning = false
        } else {
            intent.action = AmbientAudioService.ACTION_START_AMBIENT
            ContextCompat.startForegroundService(this, intent)
            isAmbientServiceRunning = true
        }
    }
}
