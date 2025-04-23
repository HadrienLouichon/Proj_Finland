# Proj_Finland
***Track updates of the code for the drone - Machine Learning project for the research at the University of Jyväskylä***

## Notion
Link to Notion : https://www.notion.so/hadrienlouichon/Stage-Finlande-1d6d0aa4a0e68065827efc70233b064a?pvs=4

## Requirements :
1. Installation of the MQTT Broker (Mosquitto) :

To allow the communication between IOT devices, the use of a Broker  is needed. I chose Mosquito Brocker, download here : https://mosquitto.org/

Then, execute the installer and follow the basic instructions. Keep track of the path where it has been installed (we will need this path for the [distant communication](#launch-distant-communication-through-wifi)).

Finally, you might have to add Mosquitto to the Windows PATH :

Open Start menu --> search "Modify system environment variables" --> Press "Environment Variables" --> Under "System Variables", find the **Path** variable and click "Edit" --> Finally click "New" and add the path :
```
C:\Program Files\mosquitto\
```

2. Installation of python libraries :

Open a CMD Terminal and install the following library using pip :
```
pip install paho-mqtt
```
In case other libraries need to be installed (this souldn't be mandatory), install them using pip :
```
pip install os-sys uuid
```
## Content Table :
- [Launch a basic local communication](#launch-basic-local-communication-branch-send_multiple_documents)
- [Launch a basic distant communication](#launch-distant-communication-through-wifi)


## Launch basic local communication (_branch Send_multiple_documents_)

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

    Then, open two different CMD terminals, one for the transmitter and one for the receiver. Launch the files using : 
    ```
    1st terminal : python receiver.py
    2nd terminal : python transmitter.py
    ```

Press **Ctrl + C** to stop the receiver program when you have succesfully received the files.

## Launch distant communication (through WiFi, _branch Simple_communication_with_RaspberryPi_) :

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