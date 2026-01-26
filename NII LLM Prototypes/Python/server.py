import socket,os
import threading
import leveled_agent_with_feedback as lv
import time

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345  
sock.bind(('0.0.0.0', port))  
sock.listen(5)

threads = []
buffer_size = 32768
cefr_levels = ["A1","A2","B1","B2", "C1", "C2"]

user_began_chatting = False

def recv_msg(connection, agent):
    global user_began_chatting
    while True:
        buf = connection.recv(buffer_size).decode('utf-8').replace('\n',' ').replace('\r','')
        print(f"Received Payload: '{buf}'")
        if (buf[0:3] == 'END'):
            print(f"Closing connection from {address}")
            break
        msg = buf[5:-1]
        response = agent.chat_to_agent(msg)
        connection.send((response + "\n").encode('utf-8'))
        user_began_chatting = True

def serve(connection, address):
    global user_began_chatting

    print(f"Connection received from {address}")
    cefr_level = -1
    intro_level = 'A1'
    phase_levels = []
    phase_times = [60, 300, 300]
    agent = False
    timer = 0

    buf = connection.recv(buffer_size).decode('utf-8').replace('\n',' ').replace('\r','')
    if (buf[0:3] == 'END'):
        print(f"Closing connection from {address}")
        connection.close()
        return
    msg = buf[5:-1]
    # TODO: response handling should be in different thread, user can spam or end during level iteration
    response = ""
    if cefr_level < 0:
        try:
            words = msg.split()
            cefr_level = cefr_levels.index(words[0])
            permutation = int(words[1])
            if permutation == 0:
                phase_levels = [intro_level, cefr_level, 'unfiltered']
            elif permutation == 1:
                phase_levels = [intro_level, 'unfiltered', cefr_level]
            else:
                raise ValueError(f"'{permutation}' is an invalid permutation value.")
        except:
            response = f"'{msg}' is in an invalid format. Try again."
            cefr_level = -1
    connection.send((response + "\n").encode('utf-8'))

    agent = lv.LeveledAgentWithFeedback(max_iter=5, min_lv_ratio=0.66, user_cefr_level=intro_level, verbosity=True)
    
    for phase in range(len(phase_levels)):
        user_began_chatting = False
        recv_thread = threading.Thread(target=recv_msg, args=(connection, agent))
        recv_thread.start()
        while True:
            if not user_began_chatting:
                continue
            print(f'Beginning of phase {phase}. Sleeping for {phase_times[phase]} seconds.')
            time.sleep(phase_times[phase])
            break
        print(f'{phase_times[phase]} seconds elapsed. End of phase {phase}.')
        recv_thread.join()
        if (phase == len(phase_levels) - 1):
            # ending transition
            connection.send('Got to go. Sorry. Bye!'.encode('utf-8'))
        else:
            connection.send('**BREAK**'.encode('utf-8'))
            phase += 1   
            agent.set_level(phase_levels[phase])
            user_began_chatting = False

    connection.close()

print("Server initialized.")
while True:  
    connection,address = sock.accept()  
    thread = threading.Thread(target=serve, args=(connection, address))
    threads.append(thread)
    thread.start()

sock.close()