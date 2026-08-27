package com.execcoach.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

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
    background = Color(0xFFF8FAFC),
    surface = Color.White,
    onPrimary = Color.White,
    onSecondary = Color.White,
    onBackground = ExecutiveNavy,
    onSurface = ExecutiveNavy
)

@Composable
fun ExecCoachTheme(
    darkTheme: Boolean = true, // Default to sleek executive dark mode
    content: @Composable () -> Unit
) {
    val colorScheme = if (darkTheme) DarkColorScheme else LightColorScheme

    MaterialTheme(
        colorScheme = colorScheme,
        content = content
    )
}
