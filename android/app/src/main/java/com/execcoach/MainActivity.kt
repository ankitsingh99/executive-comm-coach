package com.execcoach

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.view.ViewGroup
import android.webkit.PermissionRequest
import android.webkit.WebChromeClient
import android.webkit.WebResourceRequest
import android.webkit.WebResourceResponse
import android.webkit.WebSettings
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.core.view.WindowCompat
import androidx.webkit.WebViewAssetLoader
import androidx.webkit.WebViewAssetLoader.AssetsPathHandler
import dagger.hilt.android.AndroidEntryPoint

import androidx.activity.OnBackPressedCallback
import android.widget.Toast

@AndroidEntryPoint
class MainActivity : ComponentActivity() {

    private var pendingPermissionRequest: PermissionRequest? = null
    private var webViewInstance: WebView? = null
    private var backPressedTime = 0L
    private var isDarkThemeState by mutableStateOf(true)

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        val recordAudioGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false
        runOnUiThread {
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
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Enable edge-to-edge window decor with Compose statusBarsPadding & navigationBarsPadding
        // to strictly prevent the app HUD from overspilling into the notification bar / camera notch.
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val nightModeFlags = resources.configuration.uiMode and android.content.res.Configuration.UI_MODE_NIGHT_MASK
        val isSystemDark = nightModeFlags == android.content.res.Configuration.UI_MODE_NIGHT_YES
        isDarkThemeState = isSystemDark
        updateSystemBarTheme(isSystemDark)

        // Handle Android Back gesture / button smoothly without abrupt app exit
        onBackPressedDispatcher.addCallback(this, object : OnBackPressedCallback(true) {
            override fun handleOnBackPressed() {
                val wv = webViewInstance
                if (wv != null) {
                    wv.evaluateJavascript("typeof window.handleAndroidBack === 'function' ? window.handleAndroidBack() : false") { handledStr ->
                        val handled = handledStr == "true" || handledStr == "\"true\""
                        if (!handled) {
                            if (wv.canGoBack()) {
                                wv.goBack()
                            } else {
                                val currentTime = System.currentTimeMillis()
                                if (currentTime - backPressedTime < 2000) {
                                    isEnabled = false
                                    onBackPressedDispatcher.onBackPressed()
                                } else {
                                    backPressedTime = currentTime
                                    Toast.makeText(this@MainActivity, "Press back again to exit", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    }
                } else {
                    isEnabled = false
                    onBackPressedDispatcher.onBackPressed()
                }
            }
        })

        // Pre-request microphone & notification permissions
        requestAudioPermissions()

        setContent {
            AppWebViewContainer()
        }
    }

    fun updateSystemBarTheme(isDark: Boolean) {
        runOnUiThread {
            isDarkThemeState = isDark
            val statusBarColor = if (isDark) 0xFF070A13.toInt() else 0xFFF0F4F9.toInt()
            val navBarColor = if (isDark) 0xFF0E1424.toInt() else 0xFFFFFFFF.toInt()
            window.statusBarColor = statusBarColor
            window.navigationBarColor = navBarColor

            val insetsController = WindowCompat.getInsetsController(window, window.decorView)
            insetsController.isAppearanceLightStatusBars = !isDark
            insetsController.isAppearanceLightNavigationBars = !isDark
            webViewInstance?.setBackgroundColor(statusBarColor)
        }
    }

    fun requestAudioPermissions() {
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
                    databaseEnabled = true
                    mediaPlaybackRequiresUserGesture = false
                    allowFileAccess = true
                    allowContentAccess = true
                    allowFileAccessFromFileURLs = true
                    allowUniversalAccessFromFileURLs = true
                    useWideViewPort = false
                    loadWithOverviewMode = false
                    cacheMode = WebSettings.LOAD_DEFAULT
                    mixedContentMode = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
                    textZoom = 100
                }

                val assetLoader = WebViewAssetLoader.Builder()
                    .addPathHandler("/assets/", AssetsPathHandler(context))
                    .build()

                webViewClient = object : WebViewClient() {
                    override fun shouldInterceptRequest(
                        view: WebView?,
                        request: WebResourceRequest?
                    ): WebResourceResponse? {
                        val uri = request?.url ?: return null
                        return assetLoader.shouldInterceptRequest(uri)
                    }

                    override fun shouldOverrideUrlLoading(
                        view: WebView?,
                        request: WebResourceRequest?
                    ): Boolean {
                        return false
                    }
                }

                webChromeClient = object : WebChromeClient() {
                    override fun onPermissionRequest(request: PermissionRequest) {
                        runOnUiThread {
                            val requested = request.resources
                            val isAudio = requested.contains(PermissionRequest.RESOURCE_AUDIO_CAPTURE)
                            if (isAudio) {
                                if (ContextCompat.checkSelfPermission(
                                        context,
                                        Manifest.permission.RECORD_AUDIO
                                    ) == PackageManager.PERMISSION_GRANTED
                                ) {
                                    request.grant(requested)
                                } else {
                                    pendingPermissionRequest = request
                                    permissionLauncher.launch(arrayOf(Manifest.permission.RECORD_AUDIO))
                                }
                            } else {
                                request.grant(requested)
                            }
                        }
                    }

                    override fun onJsAlert(view: WebView?, url: String?, message: String?, result: android.webkit.JsResult?): Boolean {
                        result?.confirm()
                        return true
                    }

                    override fun onJsConfirm(view: WebView?, url: String?, message: String?, result: android.webkit.JsResult?): Boolean {
                        result?.confirm()
                        return true
                    }

                    override fun onJsPrompt(view: WebView?, url: String?, message: String?, defaultValue: String?, result: android.webkit.JsPromptResult?): Boolean {
                        result?.confirm()
                        return true
                    }
                }

                // Register 100% On-Device AI Engine Bridge for zero cloud dependency
                val coachEngine = com.execcoach.ai.OnDeviceCoachEngine(context)
                addJavascriptInterface(
                    com.execcoach.ai.CoachBridgeInterface(context, coachEngine),
                    "AndroidCoachAI"
                )

                // Load via secure WebViewAssetLoader domain so navigator.mediaDevices & SpeechRecognition are natively enabled
                loadUrl("https://appassets.androidplatform.net/assets/index.html")
            }
        }

        DisposableEffect(webView) {
            webViewInstance = webView
            onDispose {
                webView.destroy()
                webViewInstance = null
            }
        }

        Box(
            modifier = Modifier
                .fillMaxSize()
                .background(if (isDarkThemeState) Color(0xFF070A13) else Color(0xFFF0F4F9))
                .statusBarsPadding()
                .navigationBarsPadding()
        ) {
            AndroidView(
                factory = { webView },
                modifier = Modifier.fillMaxSize()
            )
        }
    }


    override fun onDestroy() {
        super.onDestroy()
        webViewInstance?.destroy()
        webViewInstance = null
    }
}
