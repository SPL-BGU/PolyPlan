from envs import AdvancedMinecraft


def main():
    # close pal
    try:
        env = AdvancedMinecraft(visually=False, start_pal=False, keep_alive=False)
        env.reset()
        env.close()
        print("PAL closed successfully")
    except Exception as e:
        print("PAL is not running")


if __name__ == "__main__":
    main()
