def print_section_title(title: str) -> None:
    separator = "=" * 90

    print(f"\n{separator}")
    print(f"{title}")
    print(separator)


def print_article(
    rank: int,
    score: float,
    title: str,
    category: str | None,
    year: int | str | None,
    abstract: str,
    extra: str | None = None,
    score_label: str = "Score",
    abstract_limit: int = 200,
) -> None:
    print(f"\n{rank}. {title}")
    print(f"{score_label}: {score:.4f}")
    print(f"Category: {category}")
    print(f"Year: {year}")
    print(f"Abstract: {str(abstract)[:abstract_limit]}...")

    if extra:
        print(extra)

