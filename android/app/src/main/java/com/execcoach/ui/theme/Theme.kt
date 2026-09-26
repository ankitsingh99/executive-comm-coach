package com.execcoach.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

enum class ThemeMode {
    DARK,
    LIGHT,
    SYSTEM
}

private val DarkColorScheme = darkColorScheme(
    primary = ExecutiveAccent,
    secondary = ExecutiveEmerald,
    tertiary = ExecutiveAmber,
    background = BackgroundDark,
    surface = CardBackgroundDark,
    onPrimary = Color.Black,
    onSecondary = Color.Black,
    onBackground = TextPrimaryDark,
    onSurface = TextPrimaryDark
)

private val LightColorScheme = lightColorScheme(
    primary = ExecutiveNavy,
    secondary = ExecutiveEmerald,
    tertiary = ExecutiveAmber,
    background = Color(0xFFF0F4F9),
    surface = Color.White,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onBackground = ExecutiveNavy,
    onSurface = ExecutiveNavy
)

@Composable
fun ExecCoachTheme(
    themeMode: ThemeMode = ThemeMode.SYSTEM,
    darkTheme: Boolean = when (themeMode) {
        ThemeMode.DARK -> true
        ThemeMode.LIGHT -> false
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
    },
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}

