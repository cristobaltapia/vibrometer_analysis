echo Updating sounddevice.py ...
copy .\portaudio\sounddevice_custom.py sounddevice.py .\venv\Lib\site-packages\sounddevice.py /y
copy .\libportaudio32bit.dll .\venv\Lib\site-packages\libportaudio32bit.dll /y
copy .\libportaudio64bit.dll .\venv\Lib\site-packages\libportaudio64bit.dll /y
