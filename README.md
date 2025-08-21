# Smart Cursor - Hands-Free Mouse with Nose & Eye Control

**Smart Cursor** is a Python-based tool that turns your webcam into a
**hands-free mouse controller**.\
It uses **MediaPipe face tracking** and **eye winks** to let you move
and click the mouse without using your hands.

This project is built with the intention of **helping people
with disabilities** to interact with their computers more easily.

------------------------------------------------------------------------

##  Features

-   Move the mouse pointer using your **nose position** detected via
    webcam
-   Perform clicks using **eye winks** (left eye = drag, right eye =
    right click)
-   Scroll using nose movements
-   Overlay with status indicators
-   Configurable calibration & sensitivity
-   Works on **Windows**, **Linux**, and **macOS**

------------------------------------------------------------------------

##  Requirements

Make sure you have Python 3.9+ installed. Then install the dependencies:

``` bash
pip install opencv-python mediapipe numpy pyautogui pynput
```

------------------------------------------------------------------------

##  Running from Source

Clone the repository and run:

``` bash
python "Smart Cursor.py"
```

------------------------------------------------------------------------

##  Building Executable (Windows)

If you want a standalone `.exe` file:

1.  Install PyInstaller:

    ``` bash
    pip install pyinstaller
    ```

2.  Build executable:

    ``` powershell
    pyinstaller --onefile --windowed --name "Smart Cursor" `
      --icon="cursor(1).ico" `
      --collect-all mediapipe `
      --collect-all cv2 `
      --hidden-import "mediapipe.python._framework_bindings" `
      "Smart Cursor.py"
    ```

3.  Find your exe inside the `dist/` folder:

        dist/Smart Cursor.exe

------------------------------------------------------------------------

##  Controls (Hotkeys)

-   **Q / ESC** → Quit
-   **C** → Center calibration
-   **S** → Span wizard (auto calibration)
-   **B** → Blink calibration reset
-   **L / R** → Test left / right click
-   **D** → Switch drag behavior (hold or toggle)
-   **M** → Toggle scroll mode
-   **P** → Pause / resume cursor control
-   **O** → Toggle overlay
-   **H** → Toggle help text

------------------------------------------------------------------------

##  Contributing

This project is **free for personal use** under the [MIT
License](LICENSE).\
 **Non-commercial use only** -- please do not resell or package it
commercially.

If you have suggestions, ideas, or improvements, feel free to **open an
issue** or **contact me**.\
I built this with the intention of helping people with disabilities ---
your feedback can make it even better for them! 

------------------------------------------------------------------------

##  License

MIT License

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the
"Software"), to use, copy, modify, merge, publish, and distribute copies
of the Software, subject to the following conditions:

-   The Software is provided for **personal and non-commercial use
    only**.\
-   Commercial redistribution, sublicensing, or reselling is **not
    permitted**.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED.

------------------------------------------------------------------------

## 📧 Contact

If you use this project and it helps you --- or if you have suggestions
to improve it --- please reach out!

Together we can make technology more accessible for everyone. 🌍
