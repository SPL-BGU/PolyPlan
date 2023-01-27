from envs import BasicMinecraft


def main():
    env = BasicMinecraft(visually=True, start_pal=False, keep_alive=True)

    env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        env.render()

    env.close()


if __name__ == "__main__":
    main()
