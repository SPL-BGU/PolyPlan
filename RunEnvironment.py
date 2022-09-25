import socket, time, json, sys


def main():
    domain_location = (
        "RESET domain /home/benjamin/Downloads/pal/available_tests/pogo_nonov.json\n"
    )
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "-domain":
        print(f"RESET domain {args[1]}\n")
    elif len(args) != 0:
        print("RunEnvironment.py -domain <domain_location>")
        sys.exit(2)

    print("START INITIALIZING")

    AGENT_HOST = "127.0.0.1"
    AGENT_PORT = 9000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((AGENT_HOST, AGENT_PORT))

    BUFF_SIZE = 4096
    time.sleep(1)

    sock.send(str.encode("START\n"))
    part = sock.recv(BUFF_SIZE)
    while len(part) >= BUFF_SIZE:
        part = sock.recv(BUFF_SIZE)
    time.sleep(1)

    sock.send(str.encode(domain_location))
    part = sock.recv(BUFF_SIZE)
    while len(part) >= BUFF_SIZE:
        part = sock.recv(BUFF_SIZE)
    time.sleep(3)

    sock.close()

    print("Done INITIALIZING")


if __name__ == "__main__":
    main()
