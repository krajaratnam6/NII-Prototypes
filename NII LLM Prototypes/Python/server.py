import socket,os
import threading
import leveled_agent_with_feedback as lv
import time
import unicodedata
from datetime import datetime

sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345  
sock.bind(('0.0.0.0', port))  
sock.listen(5)

threads = []
buffer_size = 32768
cefr_levels = ["A1","A2","B1","B2", "C1", "C2"]

phase_times = [10, 11, 12]
intro_level = 'A1'

def log(f, msg):
    m = unicodedata.normalize('NFKD', msg).encode('ascii', 'ignore').decode('ascii')
    f.write(m + "\n")
    print(m)

def serve(connection, address, agent):
    filename = "logs/" + str(datetime.now()).split('.',1)[0].replace(':','-') + ".txt"
    file = open(filename, 'w')
    log(file, f"Connection received from {address}")
    agent.set_file(file)
    cefr_level = -1
    phase_levels = []
    phase = 0

    while True:
        buf = connection.recv(buffer_size).decode('utf-8').replace('\n',' ').replace('\r','')
        if (buf[0:3] == 'END'):
            log(file, f"END received. Closing connection from {address}")
            break

        if (buf[0:3] == 'LOG'):
            log(file, buf[5:-1])
            continue

        if (buf[0:5] == 'PHASE'):
            phase += 1
            if (phase < len(phase_levels)):
                log(file, f"\n\n***END OF PHASE {phase - 1}*** Setting agent filter to level: {phase_levels[phase]}")
                agent.set_level(phase_levels[phase])
                continue
            else:
                log(file, f"\n\n**END OF PHASE {phase}*** Preparing outro.")
                response = agent.chat_to_agent("", ending=True)
                response = unicodedata.normalize('NFKD', response).encode('ascii', 'ignore').decode('ascii')
                connection.send((response + "\n").encode('utf-8'))
                continue
            
        if (buf[0:7] == 'SILENCE'):
            log(file, f"\n\nReceived Message: <user silence>\n\n")
            response = agent.chat_to_agent("", silence=True)
            response = unicodedata.normalize('NFKD', response).encode('ascii', 'ignore').decode('ascii')
            connection.send((response + "\n").encode('utf-8'))
            continue
            
        msg = buf[5:-1]
        # TODO: response handling should be in different thread, user can spam or end during level iteration
        response = ""
        if cefr_level < 0:
            try:
                words = msg.split()
                target_level = words[0]
                if (words[0] == "A0"): # clamping target level to be [A1,A2]
                    target_level = "A1"
                elif (words[0] == "B1"):
                    target_level = "A2"
                cefr_level = cefr_levels.index(target_level)
                permutation = int(words[1])
                if permutation == 0:
                    phase_levels = [intro_level, target_level, 'unfiltered']
                elif permutation == 1:
                    phase_levels = [intro_level, 'unfiltered', target_level]
                else:
                    raise ValueError(f"'{permutation}' is an invalid permutation value.")
                log(file, f"Declared level: {words[0]}. Target Level: {target_level}. Permutation: {words[1]}")
                log(file, f"Conversational introduction (Phase 0) will be leveled to: {phase_levels[0]}")
                for i in range(1,len(phase_levels)):
                    log(file, f"Phase {i} will be filtered to: {phase_levels[i]}")
                phase = 0
                agent.set_level(phase_levels[phase])
                response = "Success"
            except:
                response = f"'{msg}' is in an invalid format. Try again."
                cefr_level = -1
        else:
            log(file, f"\n\nReceived Message: '{msg}'\n\n")
            response = agent.chat_to_agent(msg)
            response = unicodedata.normalize('NFKD', response).encode('ascii', 'ignore').decode('ascii')

        connection.send((response + "\n").encode('utf-8'))
    connection.close()

def sock_accept():
    while True:    
        print("Initializing new agent.")
        agent = lv.LeveledAgentWithFeedback(max_iter=5, min_lv_ratio=0.66, user_cefr_level=intro_level, verbosity=True)
        agent.chat_to_agent("...", testing=True)
        lv.level_ratio("testing 1 2 3", 0)
        print("Agent initialized. Ready to accept new connections.")
        connection,address = sock.accept()
        print(f"Connection received from {address}.")
        thread = threading.Thread(target=serve, args=(connection, address, agent))
        threads.append(thread)
        thread.start()
    sock.close()

sock_accept()