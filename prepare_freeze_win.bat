echo Updating sounddevice.py ...
copy .\portaudio\sounddevice_custom.py sounddevice.py .\venv\Lib\site-packages\sounddevice.py /y
copy .\portaudio\libportaudio32bit.dll .\venv\Lib\site-packages\bin\libportaudio32bit.dll /y
copy .\portaudio\libportaudio64bit.dll .\venv\Lib\site-packages\bin\libportaudio64bit.dll /y
