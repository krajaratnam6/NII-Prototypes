import socket,os
import threading
import leveled_agent_with_feedback as lv
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
port = 12345  
sock.bind(('0.0.0.0', port))  
sock.listen(5)

threads = []
buffer_size = 32768
cefr_levels = ["A1","A2","B1","B2", "C1", "C2"]

def serve(connection, address):
    print(f"Connection received from {address}")
    cefr_level = -1
    agent = False
    while True:
        buf = connection.recv(buffer_size).decode('utf-8').replace('\n',' ').replace('\r','')
        if (buf[0:3] == 'END'):
            break
        msg = buf[5:-1]
        response = ""
        if cefr_level < 0:
            try:
                lvl = cefr_levels.index(msg.upper())
                cefr_level = lvl
            except:
                response = f"'{msg}' is not a recognized CEFR level. Try again."
        else:
            if not agent:
                agent = lv.LeveledAgentWithFeedback(max_iter=5, min_lv_ratio=0.66, user_cefr_level=cefr_level, verbosity=False)
            response = agent.chat_to_agent(msg)
        connection.send((response + "\n").encode('utf-8'))
    connection.close()

while True:  
    connection,address = sock.accept()  
    thread = threading.Thread(target=serve, args=(connection, address))
    threads.append(thread)
    thread.start()

sock.close()