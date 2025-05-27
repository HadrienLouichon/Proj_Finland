# Proj_Finland
***Track updates of the code for the drone - Machine Learning project for the research at the University of Jyväskylä***

## Notion
Link to Notion : https://www.notion.so/hadrienlouichon/Stage-Finlande-1d6d0aa4a0e68065827efc70233b064a?pvs=4

## Requirements :

### On Windows :

1. Installation of the MQTT Broker (Mosquitto) :

To allow the communication between IOT devices, the use of a Broker  is needed. I chose Mosquito Brocker, download here : https://mosquitto.org/
The documentation can be found here : https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html

Then, execute the installer and follow the basic instructions. Keep track of the path where it has been installed (we will need this path for the [distant communication](#launch-distant-communication-through-wifi)).

Finally, you might have to add Mosquitto to the Windows PATH :

Open Start menu --> search "Modify system environment variables" --> Press "Environment Variables" --> Under "System Variables", find the **Path** variable and click "Edit" --> Finally click "New" and add the path :
```
C:\Program Files\mosquitto\
```

2. Installation of python libraries :

On your computer :
Open a CMD Terminal and install the following library using pip :
```
pip install paho-mqtt
```
In case other libraries need to be installed (this souldn't be mandatory), install them using pip :
```
pip install os-sys uuid
```

### On Linux / Ubuntu :

1. Installation of the MQTT Broker (Mosquitto) :

    Go to a terminal and install Mosquitto MQTT Broker using the following commands :
    ```
    sudo apt update
    sudo apt install mosquitto mosquitto-clients
    ```
    You can also install net-tools to check if the ports are open (when mosquitto is runnin) :
    ```
    sudo apt install net-tools
    ```
    Check if the installation succeded by testing Mosquitto using 3 different terminals :
    ```
    1st Terminal : mosquitto        #To launch mosquitto
    2nd Terminal : mosquitto_sub -t "test/topic" -v
    3rd Terminal : mosquitto_pub -t "test/topic" -m "hello"
    ```
    You should now receive "hello" in your second terminal ! If not, try reinstalling Mosquitto, or check its status using :
    ```
    sudo systemctl status mosquitto
    ```

2. Installation of python and its libraries :

- If you don't want to use Jupyter :
    On your computer :
    Open a Terminal and first install at least python 3.13 using the following (you can first check the current version with *python*):
    ```
    pip install paho-mqtt
    ```
    In case other libraries need to be installed (this souldn't be mandatory), install them using pip :
    ```
    pip install os-sys uuid
    ```
- If you plan to use Jupyter :
    You may need to use a virtual environment, to install the librarires and execute the code directly from Jupyter. To do this, you first need to install venv in a Terminal :
    ```
    sudo apt install python3.8-venv
    ```
    Then go to the folder that contains the code and initialize a virtual environment : 
    ```
    cd #Name_Of_Your_Folder
    python3 -m venv venv
    source venv/bin/activate
    pip install ipykernel
    ```

### On the RaspberryPi's :

Open a Terminal and install the following library using sudo (if you already have at least python 3.13, otherwise install it with the same command lines as [here](#on-linux--ubuntu)) :
```
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.13
```
Once installed, you may have to set it so that each time you log into the PC, the newest version is running. Check it with the command *python*.If the wrong version is up, add python versions to the alternative system using : 
```
sudo update-alternatives --install /usr/bin/python python /usr/bin/python2.7 1 [Priority]
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2
```
Then check the configuration using :
```
sudo update-alternatives --config python
```
See if the status 'auto mode' is set with the correct python version !


## Content Table :
- [Launch a basic local communication](#launch-basic-local-communication-branch-send_multiple_documents--send_heavier_data)
- [Launch a basic distant communication](#launch-distant-communication-through-wifi-branch-simple_communication_with_raspberrypi--large_communication_with_raspberrypi-)
- [Launch a basic communication using different network](#launch-distant-communication-with-different-network-connexions-through-wifi-branch-simple_communication_with_raspberrypi_online_mqtt)
- [Launch a RaspberryPi - Ubuntu Computer communication](#launch-a-raspberrypi---ubuntu-computer-communication-through-same-wifi-branch-main)
- [Reset & Reinstalling a RaspberryPi 5](#reset-and-reinstalling-a-raspberrypi-5-from-scratch)

## Caracteristics of each branch :
- Branch *main* : Last up to date branch, which will be used for the communication between the RaspberryPi's and the computer.
- Branch *First_impl_Simple_Local_Transmitter_Receiver* : First functionnal implementation of the Mosquitto MQTT code, transmitting a small data in local.
- Branch *Send_multiple_documents* : Second functionnal implementation of the MQTT, transmitting multiple documents / data locally.
- Branch *Send_heavier_data* : Allows to transmit multiple large set data by decomposing it into chunks.
- Branch *Simple_communication_with_RaspberryPi* : First functionnal implementation of online MQTT, transmitting multiple documents from two different devices by being on the same network.
- Branch *Simple_communication_with_RaspberryPi_Online_MQTT* : Second functionnal implementation of online MQTT, using a remote MQTT (test.mosquitto.org).
- Branch *Large_communication_with_RaspberryPi* : Allows to send large documents / data between different devices.
  
## Launch basic local communication (_branch Send_multiple_documents & Send_heavier_data_)

1. Start Mosquitto MQTT Broker :

    Open a **Windows Powershell with the administrator rights** (Press Windows --> Search for *Windows PowerShell --> Execute it with Administrator rights).

    Close any existing mosquitto broker with : 
    ```
    net stop mosquitto
    ```
    Start a new mosquitto broker with : 
    ```
    net start mosquitto
    ```

    To check if everything worked well, open a regular CMD Terminal and check if you have the following when writing this :
    ```
    Input : netstat -an | findstr 1883
    What you need to see : TCP   127.0.0.1:1883   0.0.0.0:0   LISTENING
    ```

2. Launch the program :

    Firstly, go to the working repository and put the files you wish to transmit in the **files_to_send** repository.

    Then, open multiple different CMD terminals (one for each transmitter and one for the receiver). Launch the files using : 
    ```
    1st terminal : python receiver.py
    2nd terminal : python transmitter.py
    3rd terminal : python transmitter2.py
    ...
    ```

Press **Ctrl + C** to stop the receiver program when you have succesfully received the files.

## Launch distant communication (through WiFi, _branch Simple_communication_with_RaspberryPi & Large_communication_with_RaspberryPi_) :

1. Open a Listening point to communicate through :

    When installing Mosquitto for the first time (if you haven't done it yet, click [here](#requirements)), it will set the listening PORT to *localhost 1883*, which only works if you want to transmit datas locally (on the same machine). So you need to change this setting in order to allow distant communication :

    Go to your mosquitto repository (where mosquitto was installed on your PC). On Windows, it's typically **C:\Program Files\mosquitto**.
    - Create a new configuration file, named for example ***mosquitto_custom.conf*** (you may need administrator rights to do so) and open it with an editor (also with the administrator rights).
    Write this on the file :
        ```
        listener 1883
        allow_anonymous true
        ```
    - Or you can download the file ***mosquitto_custom.conf*** from the *setup* repository and place it in the mosquitto repository.

2. Start the Mosquitto MQTT Broker :

    You can start the Mosquitto MQTT Broker by executing the file **Mosquitto_MQTT.bat** from the folder *Setup*. It will work if Mosquitto was installed here : "C:\Program Files\mosquitto\mosquitto_custom.conf", else you can adapt it with your own PATH.

    If it doesn't work :
    
    Open a **Windows Powershell with the administrator rights** (Press Windows --> Search for *Windows PowerShell --> Execute it with Administrator rights).

    Close any existing mosquitto broker with : 
    ```
    net stop mosquitto
    ```
    Then launch a new one, with the name of your custom configuration (check previous step if you are unfamiliar with it) using :
    ```
    mosquitto -v -c "C:\Program Files\mosquitto\mosquitto_custom.conf"
    ```

    Keep this shell always open during the transmission. To check if everything worked well, open a regular CMD Terminal and check if you have the following when writing this :
    ```
    Input : netstat -an | findstr 1883
    What you need to see : TCP   0.0.0.0:1883   0.0.0.0:0   LISTENING
    ```

    If this doesn't show, try to reexecute the previous steps. You may also need to tell your firewall to open the port 1883 in your Administrative PowerShell using : 
    ```
    netsh advfirewall firewall add rule name="Mosquitto MQTT" dir=in action=allow protocol=TCP localport=1883
    ```

3. Start the communication :

- On the receiver side : Connect the device to Internet and look for its IP Adress (--> IPV4 when you try *ipconfig* in a CMD terminal). Open a CMD Terminal and execute the **receiver.py** file using : 
    ```
    python receiver.py
    ```

- On the RaspberryPi side : Make sure the RaspberryPi is connected with WiFi. Then, put the IP adress of the receiver (MQTT Broket, for example your computer or a distant server) in the **transmitter.py** file (line _MQTT_Broker_).

    On the RaspberryPi : Go to the repository that contains your code (using ls / cd), and put the files you wish to send in the repository (files_to_send). You can then launch the python code using :
    ```
    python transmitter.py
    ```
    
Press **Ctrl + C** to stop the receiver program when you have succesfully received the files. The data should have been send to the receiver with success !

## Launch distant communication with different network connexions (through WiFi, _branch Simple_communication_with_RaspberryPi_Online_MQTT_) :

1. Modify the MQTT_Broker parameter in the python files

    Open both programs **transmitter.py** & **receiver.py** and modify the line _MQTT_Broker_ to connect to the online MQTT_Broker *test.mosquitto.org* :
    ```
    MQTT_BROKER = "test.mosquitto.org"
    ```
    You should also change the name of the topics, to avoid any conflicts with other users (because the broker is accessible to anyone), so don't mind changing both line ***in both files*** as follows (here's an example) :
    ```
    TOPIC_IMAGE = "pi/photo_proj_finland"
    TOPIC_ACK = "pc/ack_proj_finland"
    ```

2. Start the communication :

    - On the receiver side : Connect the device to Internet, open a CMD Terminal and execute the **receiver.py** file using : 
        ```
        python receiver.py
        ```

    - On the RaspberryPi side : Make sure the RaspberryPi is connected with WiFi.

        Go to the repository that contains your code (using ls / cd), and put the files you wish to send in the repository (files_to_send). You can then launch the python code using :
        ```
        python transmitter.py
        ```
        Or you can launch the python file from the bash file :
        ```
        bash transmitter.py
        ```
Press **Ctrl + C** to stop the receiver program when you have succesfully received the files. The data should have been send to the receiver with success !

## Launch a RaspberryPi - Ubuntu Computer communication (through same WiFi, _branch main_) :

1. Modify the MQTT_Broker parameter in the python files :

    Open a Terminal in your computer, execute this command and copy the IP Adress shown :
    ```
    hostname -I
    ```
    Then open both programs **transmitter.py** & **receiver.py** and modify the line _MQTT_Broker_ with the previous IP Adress. You can also stop all already existing mosquitto connexions with :
    ```
    sudo systemctl stop mosquitto
    sudo netstat -tulpn | grep 1883     #Check if anything shows.
    ```

2. Start the communication :

- On the computer : Go to the **Setup** folder in a Terminal and start Mosquitto using (let this terminal always open): 
    ```
    chmod +x Mosquitto_MQTT.sh
    bash ./Mosquitto_MQTT.sh
    ```
    If there's an issue try :
    ```
    chmod +x Mosquitto_MQTT_admin.sh
    sudo ./Mosquitto_MQTT_admin.sh
    ```

    Now you can start the receiving code. 
    - If you are using Jupyter :
        You can execute all cells from the file 'receiver.ipynb', if you are asked to choose a kernel, select the one you have previously initialised (venv/bin/python), if you don't know what this is about, check this [section](#on-linux--ubuntu)
    
     by opening another Terminal (while still being in the **Setup** folder):
    ```
    bash ./Receiver.sh
    ```

- On the RaspberryPi side : Make sure the RaspberryPi is connected with the same WiFi as the computer. Put all the files you wish to transmit in the folder *files_to_send*, in the folder *Code_Mosquitto/RaspberryPi*.

    Open a terminal and go to the same **Setup** folder and execute :
    ```
    bash Transmitting.sh
    ```
Press **Ctrl + C** to stop the receiver program & the Mosquitto when you have succesfully received the files and want to end the transmission. The data should have been send to the receiver with success (in the folder Code_Mosquitto/MQTT_Broker/files_received)!

## Reset and Reinstalling a RaspberryPi 5 from scratch :

- Make a clone of the Raspberry to an external USB Drive in case something wrong happen :
    Go to a terminal in your Raspberry and do (you need to have GIT, otherwise install it):
    ```
    git clone https://github.com/billw2/rpi-clone.git 
	cd rpi-clone
	sudo cp rpi-clone rpi-clone-setup /usr/local/sbin
    ```
    Insert your USB Drive, and look for its PATH (ex ; /dev/sda) using :
    ```
    lsblk
    ```
    You may need to unmount the destination, so look for PATHs like */dev/sda1*, */dev/sda2*, ... and do for each one of them :
    ```
    sudo umount #Put_the_PATH_here 
    ex : sudo umount /dev/sda1
    ...
    ```
    Then so this (take only the last part of the PATH, for example if the PATH is **/dev/sda**, take only **sda**):
    ```
    sudo rpi-clone #terminology -v 
    ex : sudo rpi-clone sda -v
    ```
    They can ask you to choose an "Optional destination name", which will be the name of the clone, so you can put the name you wish.

- Reset the Raspberry :
    You will need to install **rpi-imager** :
    ```
    sudo apt update
    sudo apt full-upgrade
    sudo apt install rpi-imager
    sudo rpi-imager
    ```
    A window should then launch, and you can select the device (for us it is a RaspberryPi 5), its OS (the recommended one) and the Storage (must be an external one).
    
    I used this step in order to reset both of the RaspberryPis : 
    
    1. I did this previous step with a USB Driver as storage, so that a fresh Raspberry is installed on it. I then booted on this driver (shutdown the Raspberry, remove its internal SD Card, put the USB Driver and boot).
    2. After I booted, I initialised the new system so that the RaspberryPi is functional (still on the USB Drive). I then reinstalled rpi-imager and this time I used the SD Card as the Storage (you may also need to unmount the SD Card with the previous steps).
    3. You can then shutdown the RaspberryPi and boot on the SD Card to have a freshly new OS.


- Reinstalling key components :
    Just after you finished the reset with the clean and updated OS, the main features should be already installed : Python with some libraries (Numpy, Scikit), Git, Web Explorer such as Firefox.
    You can make sure your system is updated with :
    ```
    sudo apt update
    sudo apt upgrade -y
    ```
    So you'll need to install a few more components in order to make the system work :
    Additional python libraries :
    ```
    sudo apt install python3-matplotlib
    sudo apt install python3-paho-mqtt
    ```
    VS Code :
    ```
    sudo apt install code -y
    ```
    This should be it !