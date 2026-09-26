package com.execcoach

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import dagger.hilt.android.AndroidEntryPoint

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private var pendingPermissionRequest: PermissionRequest? = null
    private var webViewInstance: WebView? = null

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val recordAudioGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
        if (recordAudioGranted) {
            pendingPermissionRequest?.let { req ->
                req.grant(req.resources)
                pendingPermissionRequest = null
            }
        } else {
            pendingPermissionRequest?.deny()
            pendingPermissionRequest = null
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Ensure system status & nav bars match dark background theme
        WindowCompat.setDecorFitsSystemWindows(window, true)
        window.statusBarColor = 0xFF070A13.toInt()
        window.navigationBarColor = 0xFF0E1424.toInt()

        // Pre-request microphone permission on Android 10+
        requestAudioPermissions()

        setContent {
            AppWebViewContainer()
        }
    }

    private fun requestAudioPermissions() {
        val permissions = mutableListOf(Manifest.permission.RECORD_AUDIO)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            permissions.add(Manifest.permission.POST_NOTIFICATIONS)
        }
        val needsRequest = permissions.any {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (needsRequest) {
            permissionLauncher.launch(permissions.toTypedArray())
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    @Composable
    private fun AppWebViewContainer() {
        val context = this
        val webView = remember {
            WebView(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                setBackgroundColor(0xFF070A13.toInt())
                isVerticalScrollBarEnabled = false
                isHorizontalScrollBarEnabled = false

                settings.apply {
                    javaScriptEnabled = true
                    domStorageEnabled = true
                    mediaPlaybackRequiresUserGesture = false
                    allowFileAccess = true
                    allowContentAccess = true
                    // Ensure accurate mobile viewport scaling for Google Pixel
                    useWideViewPort = false
                    loadWithOverviewMode = false
                    cacheMode = WebSettings.LOAD_DEFAULT
                    mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                    textZoom = 100
                }

                webViewClient = object : WebViewClient() {
                    override fun shouldOverrideUrlLoading(view: WebView?, url: String?): Boolean {
                        return false
                    }
                }

                webChromeClient = object : WebChromeClient() {
                    override fun onPermissionRequest(request: PermissionRequest) {
                        runOnUiThread {
                            val isAudio = request.resources.contains(PermissionRequest.RESOURCE_AUDIO_CAPTURE)
                            if (isAudio) {
                                if (ContextCompat.checkSelfPermission(
                                        context,
                                        Manifest.permission.RECORD_AUDIO
                                    ) == PackageManager.PERMISSION_GRANTED
                                ) {
                                    request.grant(request.resources)
                                } else {
                                    pendingPermissionRequest = request
                                    permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
                                }
                            } else {
                                request.grant(request.resources)
                            }
                        }
                    }
                }

                // Register 100% On-Device AI Engine Bridge for zero cloud dependency
                val coachEngine = com.execcoach.ai.OnDeviceCoachEngine(context)
                addJavascriptInterface(
                    com.execcoach.ai.CoachBridgeInterface(context, coachEngine),
                    "AndroidCoachAI"
                )

                loadUrl("file:///android_asset/index.html")
            }
        }

        DisposableEffect(webView) {
            webViewInstance = webView
            onDispose {
                webView.destroy()
                webViewInstance = null
            }
        }

        AndroidView(
            factory = { webView },
            modifier = Modifier
                .fillMaxSize()
                .background(Color(0xFF070A13))
        )
    }

    override fun onDestroy() {
        super.onDestroy()
        webViewInstance?.destroy()
        webViewInstance = null
    }
}
