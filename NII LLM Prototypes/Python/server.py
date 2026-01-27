import socket,os
import threading
import leveled_agent_with_feedback as lv
import time
import unicodedata

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345  
sock.bind(('0.0.0.0', port))  
sock.listen(5)

threads = []
buffer_size = 32768
cefr_levels = ["A1","A2","B1","B2", "C1", "C2"]

phase_times = [10, 11, 12]
intro_level = 'A1'

user_began_chatting = False

def recv_msg(connection, agent, stop_event):
    global user_began_chatting
    while True:
        buf = connection.recv(buffer_size).decode('utf-8').replace('\n',' ').replace('\r','')
        print(f"Received Payload: '{buf}'")
        if (buf[0:3] == 'END'):
            print(f"Closing connection from {address}")
            connection.close()
            agent.ended = True
            break
        msg = buf[5:-1]
        print(f"Chatting to agent: '{buf}'")
        response = agent.chat_to_agent(msg)
        print(f"Received response: '{response}'")
        if stop_event.is_set():
            break
        response = unicodedata.normalize('NFKD', response).encode('ascii', 'ignore').decode('ascii')
        connection.send((response + "\n").encode('utf-8'))
        user_began_chatting = True

def serve(connection, address, agent):
    global user_began_chatting

    print(f"Connection received from {address}")
    cefr_level = -1
    phase_levels = []
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
                phase_levels = [intro_level, words[0], 'unfiltered']
            elif permutation == 1:
                phase_levels = [intro_level, 'unfiltered', words[0]]
            else:
                raise ValueError(f"'{permutation}' is an invalid permutation value.")
        except:
            response = f"'{msg}' is in an invalid format. Try again."
            cefr_level = -1
    connection.send((response + "\n").encode('utf-8'))
    
    for phase in range(len(phase_levels)):
        user_began_chatting = False
        stop_event = threading.Event()
        recv_thread = threading.Thread(target=recv_msg, args=(connection, agent, stop_event))
        recv_thread.start()
        while True:
            if not user_began_chatting:
                continue
            print(f'Beginning of phase {phase}. Sleeping for {phase_times[phase]} seconds.')
            recv_thread.join(timeout=phase_times[phase]) #time.sleep(phase_times[phase])
            break
        stop_event.set()
        print(f'{phase_times[phase]} seconds elapsed. End of phase {phase}.')
        recv_thread.join()
        if agent.ended:
            return
        if (phase == len(phase_levels) - 1):
            # ending transition
            connection.send('Got to go. Sorry. Bye!\n'.encode('utf-8'))
        else:
            connection.send('**BREAK**\n'.encode('utf-8'))
            phase += 1   
            agent.set_level(phase_levels[phase])
            user_began_chatting = False

    connection.close()

def sock_accept():
    while True:    
        print("Initializing new agent.")
        agent = lv.LeveledAgentWithFeedback(max_iter=5, min_lv_ratio=0.66, user_cefr_level=intro_level, verbosity=True, lrfxn=custom_level_ratio)
        agent.chat_to_agent("...")
        lv.level_ratio("testing 1 2 3", 0)
        print("Agent initialized. Ready to accept new connections.")
        connection,address = sock.accept()
        print(f"Connection received from {address}.")
        thread = threading.Thread(target=serve, args=(connection, address, agent))
        threads.append(thread)
        thread.start()
    sock.close()

def custom_level_ratio(text, lvl):
    while True:
        global level_event
        global answ_event
        global level_req
        global level_answ
        if level_event.is_set():
            time.sleep(0.05)
            continue
        level_req = (text, lvl)
        level_event.set()
        answ_event.wait()
        answ_event = threading.Event()
        return level_answ

        
answ_event = threading.Event()
level_event = threading.Event()
level_req = ""
level_answ = -1

thread = threading.Thread(target=sock_accept)
threads.append(thread)
thread.start()

while True:
    level_event.wait()
    level_answ = lv.level_ratio(level_req[0], level_req[1])
    answ_event.set()
    level_event = threading.Event()