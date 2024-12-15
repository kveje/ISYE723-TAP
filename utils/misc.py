def show_name(actor_type: str):
    if actor_type.lower()[:3] == "ucb":
        return "UCB: " + r"$\beta =$" + f" {actor_type[4:]}"
    elif actor_type.lower() == "random":
        return "Random"
    elif actor_type.lower() == "thompson":
        return "Thompson"
    else:
        return actor_type
