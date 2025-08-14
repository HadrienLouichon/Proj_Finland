# Proj_Finland
***Track updates of the code for the drone - Machine Learning project for the research at the University of Jyväskylä***

## Requirements :

### On Windows :

1. Installation of the MQTT Broker (Mosquitto) :

    To allow the communication between IOT devices, the use of a Broker  is needed. I chose Mosquito Brocker, download here : https://mosquitto.org/
    The documentation can be found here : https://eclipse.dev/paho/files/paho.mqtt.python/html/client.html

    Then, execute the installer and follow the basic instructions. Keep track of the path where it has been installed (we will need this path for the [distant communication](#launch-distant-communication-through-wifi)).

    Finally, you might have to add Mosquitto to the Windows PATH :

    Open Start menu --> search "Modify system environment variables" --> Press "Environment Variables" --> Under "System Variables", find the **Path** variable and click "Edit" --> Finally click "New" and add the path (where mosquitto was installed) :
    ```
    C:\Program Files\mosquitto\     <<--- Example
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
- [Caracteristics of each branch](#caracteristics-of-each-branch)
- [Description of the Github depot](#caracteristics-of-each-branch)
--------------
- [Final : Launch Raspberry - Ubuntu Computer with MLM Algorithm](#final-:-launch-raspberry---ubuntu-computer-with-mlm-algorithm)
- [Code description](#code-description)
------------------------------------------------------------
- [Launch a basic local communication to transfer files](#launch-basic-local-communication-branch-send_multiple_documents--send_heavier_data)
- [Launch a basic distant communication to transfer files](#launch-distant-communication-through-wifi-branch-simple_communication_with_raspberrypi--large_communication_with_raspberrypi-)
- [Launch a basic communication using different network](#launch-distant-communication-with-different-network-connexions-through-wifi-branch-simple_communication_with_raspberrypi_online_mqtt)
- [Launch a RaspberryPi - Ubuntu Computer communication](#launch-a-raspberrypi---ubuntu-computer-communication-through-same-wifi-branch-main)
--------------------
- [Reset & Reinstalling a RaspberryPi 5](#reset-and-reinstalling-a-raspberrypi-5-from-scratch)

## Caracteristics of each branch :
- Branch *main* : Last up to date branch, which will be used for the communication between the RaspberryPi's and the computer.
- Branch *First_impl_Simple_Local_Transmitter_Receiver* : First functionnal implementation of the Mosquitto MQTT code, transmitting a small data in local.
- Branch *Send_multiple_documents* : Second functionnal implementation of the MQTT, transmitting multiple documents / data locally.
- Branch *Send_heavier_data* : Allows to transmit multiple large set data by decomposing it into chunks.
- Branch *Simple_communication_with_RaspberryPi* : First functionnal implementation of online MQTT, transmitting multiple documents from two different devices by being on the same network.
- Branch *Simple_communication_with_RaspberryPi_Online_MQTT* : Second functionnal implementation of online MQTT, using a remote MQTT (test.mosquitto.org).
- Branch *Large_communication_with_RaspberryPi* : Allows to send large documents / data between different devices.
- Branch *Final_step_project* : Communication between Analyser / Server with the MLM & Kalman algorithm.
  
## Description of the Github depot :
This project is organized in differents parts :

- The *data* folder contains all test data (salinas / pavia) used in the algorithms. The main algorithms focuses on the salinas data (matlab extension), and the image is split in differents parts for training and testing (numpy extension).
- The *documents* folder contains all ressources from my internship, that describes the project.
- The *ML* folder contains all versions of the machine learning algorithm used : the first implemented is a non-recursive algorithm (*MLM_non_recursive.py*), then a recursive algorithm *MLM_recursive.py*, and finally an algorithm with the Kalman filter to improve the model *mlm_functions.py & MLM_federated.ipynb*.
- The *MQTT_Broker* folder contains two codes : a simple code to receive any kind of data through the MQTT (*simple_receiver.py* & *simple_receiver.ipynb*), and the final code containing the Kalman filter algorithm part (*Model_Upgrader.py* & *Model_Upgrader.ipynb*). It also contains the folder *files_received*, where all files from the simple receiver will be stored.
- The *RaspberryPi* folder contains two codes : a simple code to receive any kind of data through the MQTT (*simple_transmitter.py* & *simple_transmitter.ipynb*), and the final code containing the MLM algorithm part (*Image_Analyser.py* & *Image_Analyser.ipynb*). It also contains the folder *files_to_send*, where all files stored in it will be send with the simple transmitter.
- The *Setup* folder contains bash scripts to launch Mosquitto with a custom configuration, and to launch the simple receiver and transmitter.

To sum up what you absolutely need to have in order to make it work : One or two devices, one to receive and one to transmit (can be the same if local transmission). They have to be on the same network or to use an online MQTT Broker. Mosquitto has to be running on the receiver's side (if local, on 127.0.0.X, port 1883 as default ; otherwise on 0.0.0.0, port 1883 as default). Both the receiver and the transmitter's algorithms need to have the IP adress and the port of the receiver in the MQTT parameters.

## Final : Launch Raspberry - Ubuntu Computer with MLM Algorithm.

For this, you'll need to have mosquitto installed and configured on the Computer that will have to update the MLM model. Check [Requirements](#requirements) if necessary.

1. Using Terminals :
    
    I.  On the Ubuntu Computer : Open 3 different terminals, and connect to the Wifi that will be shared with the Raspberry. The first terminal will be used to watch how Mosquitto is acting, the second one will be to launch the MQTT Broker, and the last one to launch the Model_Upgrader python file.
    
    First get the IP adress of the computer using, and put it both in the 
    **Model_Upgrader.py** and the **Image_Analyser.py** file, at the line *MQTT_Broker = ""* :
    ```
    hostname -I
    ```
    Check if mosquitto isn't already working in the background using :
    ```
    sudo netstat -tulpn | grep 1883
    ```
     If nothing shows, it's good, otherwise do :
     ```
     sudo systemctl stop mosquitto
     ```

     On the second terminal, go to the **Setup** folder, and execute *Mosquitto_MQTT.sh* to launch a modified version of Mosquitto using :
     ```
     chmod +x Mosquitto_MQTT.sh
     bash ./Mosquitto_MQTT.sh
     ```
     If this doesn't work, try with the *Mosquitto_MQTT_admin.sh* one.

     On the last terminal, go to the **MQTT_Broker** folder, and launch the *Model_Upgrader.py* file (it's an infinite loop, so you can leave it open, it will improve all incoming data from any Raspberry).

     To do that, you may need to start a virtual python environment :
     ```
     python3 -m venv venv
     source venv/bin/activate
     ```
     If needed :
     ```
     pip install ipykernel
     pip install numpy
     ```

     II. On the Raspberry(s), open one terminal and go to the *RaspberryPi* folder. You first need to be connected to the same Wifi as the Ubuntu Computer, and you'll need to put its IP Adress in **Image_Analyser.py** file, at the line *MQTT_Broker = ""*.
     You can now start the analyse of the image with :
     ```
     python Image_Analyser.py
     ```
2. Using Jupyter Notebook :

## Code description :

This section will try to explain at best how to use the code and to change some of the parameters. This will only explain the files **Model_Upgrader.py** and **Image_Analyser.py**.

- Image_Analyser.py :
    This python file is analysing data row by row first by creating a MLM model, that will be sent to another file in order to be improved. It will then analyse the following one, with the newest upgraded model each time. 
    
    To do so, I used the python library paho-mqtt, that allows to communicate through a MQTT (with Wifi). After the imports, the first section creates all global variables used in the code :

    ```
    MQTT_BROKER = "#IP Adress of the MQTT Host"
    MQTT_PORT = 1883
    USE_TLS = False

    TOPIC_Data = "pi/data"
    TOPIC_ACK = "pc/ack"

    CHUNK_SIZE = 10_000_000

    ack_recu = False
    message_id_en_cours = ""
    feedback_vars = {}
    lock = threading.Lock()
    ```
    The first line is used to connect to the Computer that will receive the data, you just have to be connected to the same network and enter its IP Adress in between the "".

    The default port used by mosquitto when installing is the port 1883, but this can be changed in the file *mosquitto_custom.conf* in the **Setup** folder at the line *'listener 1883'*. They just have to match, just be careful, it may cause some issues if you change it because of firewall rules (you may need to allow this new port in the firewall parameters : TCP with all connexions possible).

    The TLS is here to allow the connexion with or without having to use creditentials (to have a more secure link). This was not set here.

    The Topic Data are like bridges that links two cities : in order to have the communication, a receiver (here the pc) has to be subscribe to a topic (here the topic *pi/data*), and when a transmitter will emit data on this specific topic, it will be sent to the receiver, or to anyone subscribe at this IP adress at this topic. The second topic is for the feedback (updated variables), so here the raspberry is subscriber, and the pc will be the sender.

    The cunk size may not be very useful here, because it allows the transfer of large data (which will be seperated in smaller pieces, called chunks). You can change its value here.

    The following variables are there to enable a stable connexion : at first, the feedback is set to no (because there is no connexion yet), the id of the message is a string variable, the feedback dictionary will store all updated variables from the PC at each execution of the loop, and the lock will be there to ensure all feedback has been received before continuing on the loop (to avoid using old data).

    ```
    def on_message(client, userdata, msg):
        '''
        This function is called when a message is received from the MQTT broker. It decodes the message, checks if it is an acknowledgment for the
        current message ID, and stores the feedback variables into a global variable "feedback_vars".
        Input : MQTT Client, Userdata, Message.
        Output : None.
        '''
    return
    ```

    ```
    def encode_variable(var):
    '''
    This function encode all type of variable in base64.
    Input : Variable(s) to encode.
    Output : Encoded variable(s) in base64.
    '''
    return
    ```
    ```
    def decode_variable(b64_str):
    '''
    This function decode all type of variable of base64 type.
    Input : Variable(s) to decode.
    Output : Decoded variable(s).
    '''
    return
    ```
    ```
    def send_variables(variables, client):
    '''
    This function sends variables to the MQTT broker, in chunks, to allow transfer of large data, and waits for the feedback.
    Input : Variables to send, MQTT Client.
    Output : Boolean (True if a feedback had been received, False otherwise).
    '''
    return
    ```
    The previous function *send_variables* will publish all the data on the topic *pi/data* so that the subscriber (PC) will receive the data to upgrade. It decompose any large data into smaller pieces to avoid any overflow, and waits everytime for a feedback in order to continue.
    ```
    def wait_feedback(timeout=5):
    '''
    This function wait for the feedback, so that the program can't continue without the latest data.
    '''
    return
    ```
    ```
    def run_simulation(...)
    ...
    for row in range(H_train):
    ...
    
    ### MQTT Communication :
        
        ## Data sent to the Raspberry / PC : anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace.
        to_send = [anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace]
        Send = send_variables(to_send, client)

        ## Receive : log_det_P, gain_norm, bhat1, bhat2, P1, P2, b_fuse, P_fuse, Q, det_P_trace.
        if Send:
            fb = wait_feedback(timeout=5)
            if fb: 
                log_det_P_trace = list(fb.get("log_det_P_trace", log_det_P_trace))
                gain_norms = list(fb.get("gain_norms", gain_norms))
                bhat1 = np.array(fb.get("bhat1", bhat1))
                bhat2 = np.array(fb.get("bhat2", bhat2))
                P1 = np.array(fb.get("P1", P1))
                P2 = np.array(fb.get("P2", P2))
                b_fuse = np.array(fb.get("b_fuse", b_fuse))
                P_fuse = np.array(fb.get("P_fuse", P_fuse))
                Q = np.array(fb.get("Q", Q))
                det_P_trace = list(fb.get("det_P_trace", det_P_trace))
        else:
            print("Error : No feedback received.")
        
        models['base'].append(b_fuse)

    ...
    ```
    This part of the *run_simulation* function is being called at each row of the image that is being analyzed. All the data that is being send to the PC is stored in the variable **to_send**, some may be use only for graphs so they can be removed (be careful, if you do a modification, don't forget to change also the *Model_Upgrader.py* code !). After they've being sent, in case of a feedback, it will redefine all used variables with their latest data.

    ```
    def main():
    '''
    Main function that launches the MQTT client and starts the image analysis. It follows the first blocl of code from 'MLM_federated.ipynb'.
    '''

    ### Launch MQTT Client and subscribe to the feedback topic.
    client = mqtt.Client()
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.subscribe(TOPIC_ACK)
    client.loop_start()
    
    ...
    ### Main code here


    client.loop_stop()
    client.disconnect()
    ```
    This main function is useless for the analysing part of this algorithm. However, it's here to load the image data from the **data** file and to start the MLM algorithm. This function can be removed / rewrite almost entirely. To ensure the MQTT connexion will work, you still need the previous code lines. This allows the connexion to the MQTT, at the correct IP adress and port ; the subscription to the feedback topic ; the deconnexion at the end. You will need these lines to make te algorithm work.

- Model_Upgrader.py :
    
    I won't come back to each MQTT functions, some of them are the same as in the *Image_Analyser.py* code. Let's look at the differences :
    ```
    buffers = {}
    received_vars = {}
    ``` 
    The *buffers* global variables is used in case of large data transfer : this will allow to reconstruct the decomposed data into the orignial one. The *received_vars* variable will store all transmitted data from the Raspberry to be upgrade with the Kalman functions.
    ```
    def on_connect(client, userdata, flags, rc):
    '''
    This function subscribe to the main topic to receive the data from the transmitter.
    '''
    print("[RECEIVER] Connected, waiting for data ...")
    client.subscribe(TOPIC_Data)
    ```
    This function subscribes to the *pi/data* topic to receive the data from the Raspberry.
    ```
    def on_message(client, userdata, msg):
    '''
    This function is called when a message is received from the MQTT broker. It decodes the message and reconstruct any potential chunks,
    and sends a feedback to the sender.
    Input : MQTT Client, Userdata, Message.
    Output : None.
    '''

    ...

            ## Received data from the transmitter : anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace.
            anomscore1 = np.array(decode_variable(vars_payload["var1"]))
            anomscore2 = np.array(decode_variable(vars_payload["var2"]))
            b_fuse     = np.array(decode_variable(vars_payload["var3"]))
            P_fuse     = np.array(decode_variable(vars_payload["var4"]))
            Q          = np.array(decode_variable(vars_payload["var5"]))
            bhat1      = np.array(decode_variable(vars_payload["var6"]))
            bhat2      = np.array(decode_variable(vars_payload["var7"]))
            P1         = np.array(decode_variable(vars_payload["var8"]))
            P2         = np.array(decode_variable(vars_payload["var9"]))
            gain_norms = list(decode_variable(vars_payload["var10"]))
            log_det_P_trace = list(decode_variable(vars_payload["var11"]))
            det_P_trace = list(decode_variable(vars_payload["var12"]))
            
            ## Upgrade the model using the Kalman Filter with the received variables.
            log_det_P_trace, gain_norms, bhat1, bhat2, P1, P2, b_fuse, P_fuse, Q, det_P_trace = upgrading_model(
                anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace
            )

            ## Send feedback with the updated variables : log_det_P, gain_norm, bhat1, bhat2, P1, P2, b_fuse, P_fuse, Q, det_P_trace.
            feedback_data = {
                "log_det_P_trace": encode_variable(log_det_P_trace),
                "gain_norms": encode_variable(gain_norms),
                "bhat1": encode_variable(bhat1),
                "bhat2": encode_variable(bhat2),
                "P1": encode_variable(P1),
                "P2": encode_variable(P2),
                "b_fuse": encode_variable(b_fuse),
                "P_fuse": encode_variable(P_fuse),
                "Q": encode_variable(Q),
                "det_P_trace": encode_variable(det_P_trace)
            }
    ...
    ```
    This function decodes all incoming messages, stores the received variables in the global variable *received_vars*. It will then call the main function **upgradig_model**, which will take the old variables, call all Kalman functions, and will return the newest updated variables. It will then send them back to the sender. If you want to remove / add / change one or more variables, you have to change them here, in the *upgrading_model* function, and IN THE *IMAGE_ANALYSER.PI* ALGORITHM !!
    ```
    def upgrading_model(anomscore1, anomscore2, b_fuse, P_fuse, Q, bhat1, bhat2, P1, P2, gain_norms, log_det_P_trace, det_P_trace):
    '''
    This function updates the model using the updated data from the MQTT Broker, and returns the updated variables.
    Input : anomscore1 (array), anomscore2 (array), b_fuse (array), P_fuse (array), Q (array), bhat1 (array), bhat2 (array), P1 (array), P2 (array), gain_norms (list), log_det_P_trace (list), det_P_trace (list)
    Output : log_det_P_trace (list), gain_norms (list), bhat1 (array), bhat2 (array), P1 (array), P2 (array), b_fuse (array), P_fuse (array), Q (array), det_P_trace (list)
    '''
    return
    ```
    This function calls each Kalman functions to have the upgraded variables. This was from the *mlm_functions.py* algorithm. You can change / add / modify as much as you want, as long as the corrections are also made in all required place !
    ```
    ### MQTT Client Setup

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()
    ```
    This part is mandatory to start the MQTT client and to subscribe to the correct topic. It uses an infinite loop so that many Raspberry could send data at the same time for an infinite amount of time.

## Launch basic local communication (_branch Send_multiple_documents & Send_heavier_data_)

This was made for a **Windows Computer** as the MQTT Broker.

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

    Firstly, go to the working repository (**RaspberryPi** repository) and put the files you wish to transmit in the **files_to_send** repository.

    Then, open multiple different CMD terminals (one for each transmitter and one for the receiver). Launch the files using : 
    ```
    1st terminal : python simple_receiver.py
    2nd terminal : python simple_transmitter.py
    3rd terminal : python simple_transmitter2.py
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
    - Or you can download the file ***mosquitto_custom.conf*** from the *Setup* repository and place it in the mosquitto repository.

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
    python simple_receiver.py
    ```

- On the RaspberryPi side : Make sure the RaspberryPi is connected with WiFi. Then, put the IP adress of the receiver (MQTT Broket, for example your computer or a distant server) in the **transmitter.py** file (line _MQTT_Broker_).

    On the RaspberryPi : Go to the repository that contains your code (using ls / cd), and put the files you wish to send in the repository (files_to_send). You can then launch the python code using :
    ```
    python simpletransmitter.py
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
        python simple_receiver.py
        ```

    - On the RaspberryPi side : Make sure the RaspberryPi is connected with WiFi.

        Go to the repository that contains your code (using ls / cd), and put the files you wish to send in the repository (files_to_send). You can then launch the python code using :
        ```
        python simple_transmitter.py
        ```
        Or you can launch the python file from the bash file :
        ```
        bash simple_transmitter.py
        ```
Press **Ctrl + C** to stop the receiver program when you have succesfully received the files. The data should have been send to the receiver with success !

## Launch a RaspberryPi - Ubuntu Computer communication (through same WiFi, _branch main_) :

1. Modify the MQTT_Broker parameter in the python files :

    Open a Terminal in your computer, execute this command and copy the IP Adress shown :
    ```
    hostname -I
    ```
    Then open both programs **simple_transmitter.py** & **simple_receiver.py** and modify the line _MQTT_Broker_ with the previous IP Adress. You can also stop all already existing mosquitto connexions with :
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
    bash Transmitter.sh
    ```
Press **Ctrl + C** to stop the receiver program & the Mosquitto when you have succesfully received the files and want to end the transmission. The data should have been send to the receiver with success (in the folder MQTT_Broker/files_received)!


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