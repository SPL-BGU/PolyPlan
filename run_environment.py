import socket, sys
import utils


def main():
    domain_location = (
        "RESET domain /home/benjamin/Downloads/pal/available_tests/pogo_nonov.json"
    )
    args = sys.argv[1:]

    if len(args) == 2 and args[0] == "-domain":
        print(f"RESET domain {args[1]}\n")
    elif len(args) != 0:
        print("RunEnvironment.py -domain <domain_location>")
        sys.exit(2)

    print("Start INITIALIZING")

    agent_host = "127.0.0.1"
    agent_port = 9000

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((agent_host, agent_port))

    data_dict = utils.send_command(sock, "START")
    print(data_dict)
    data_dict = utils.send_command(sock, domain_location)
    print(data_dict)

    sock.close()

    print("Done INITIALIZING")


if __name__ == "__main__":
    main()
