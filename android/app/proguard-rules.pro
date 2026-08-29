# Proguard rules for Executive Communication Coach

# Room & SQLCipher
-keep class androidx.room.** { *; }
-dontwarn androidx.room.**
-keep class net.sqlcipher.** { *; }
-keep class net.sqlcipher.database.** { *; }
-dontwarn net.sqlcipher.**

# Dagger Hilt
-keep class * extends androidx.activity.ComponentActivity
-keep class * extends android.app.Application
-keep class * extends android.app.Service
-keep class * extends androidx.lifecycle.ViewModel
-dontwarn com.google.errorprone.annotations.**

# ONNX Runtime & MediaPipe
-keep class ai.onnxruntime.** { *; }
-keep class com.microsoft.onnxruntime.** { *; }
-dontwarn com.microsoft.onnxruntime.**
-keep class com.google.mediapipe.** { *; }
-dontwarn com.google.mediapipe.**

# Gson & Reflection
-keepclassmembers class * {
    @com.google.gson.annotations.SerializedName <fields>;
}
-keep class com.execcoach.data.local.entity.** { *; }
