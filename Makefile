.POSIX:

apk:
	podman run -v /home/narodnik/src/stuff/miniquad:/root/mq -v $(shell pwd):/root/app -w /root/app/ -t apk cargo quad-apk build
	$(MAKE) install-apk

install-apk:
	-mv target/android-artifacts/debug/apk/mqapp-app.apk .
	-adb uninstall mqapp.mqapp_app
	adb install -r mqapp-app.apk
	reset
	adb logcat -c
	adb shell monkey -p mqapp.mqapp_app -c android.intent.category.LAUNCHER 1
	adb logcat -v color -s mqapp -s SAPP -s libc -s DEBUG -s ActivityManager -s ActivityTaskManager -s WindowManager -s AndroidRuntime -s rkfi.mqapp_app | tee output.log

fmt:
	$(CARGO) +nightly fmt

