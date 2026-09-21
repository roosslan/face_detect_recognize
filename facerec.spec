# PyInstaller spec: `pyinstaller facerec.spec` builds the single file dist/facerec.exe.
from PyInstaller.utils.hooks import collect_data_files

a = Analysis(
    ["main.py"],
    pathex=["."],
    # dlib's trained models (*.dat) are plain data files inside this package; without them the
    # exe starts fine and then fails on the first face.
    datas=collect_data_files("face_recognition_models"),
    # cv2, face_recognition and redis are imported lazily inside functions and are found by
    # the analysis; these are big packages that may be installed alongside but are not used.
    excludes=[
        "tkinter",
        "matplotlib",
        "torch",
        "torchvision",
        "scipy",
        "pandas",
        "IPython",
        "pytest",
    ],
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="facerec",
    console=True,  # logs go to the console, Ctrl+C stops the program
    upx=False,
)
