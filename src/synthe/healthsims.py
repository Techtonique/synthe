import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Optional
import warnings


class SmartHealthSimulator:
    """
    Simulates a synthetic, multimodal time series dataset resembling wearable, environmental,
    behavioral, and self-reported health data over time. Includes numeric, categorical, and text data.

    The simulator generates realistic daily records including:
    - Heart rate
    - Steps
    - Skin and ambient temperature
    - Activity label (rest, walk, exercise)
    - Mood score (1-5)
    - Mood notes (short texts)
    - Air quality index
    - Sleep quality (dependent variable)

    Includes methods for plotting time series, distributions, relationships, and text-based visualizations.

    Examples
    --------
    >>> sim = SmartHealthSimulator(days=180, seed=42)
    >>> print(sim.data.head())
    >>> sim.plot_time_series()
    >>> sim.plot_mood_sleep()
    >>> sim.plot_activity_distribution()
    >>> sim.plot_mood_wordcloud()
    """

    def __init__(self, days: int = 180, seed: int = 123):
        """
        Create a new simulator instance and generate synthetic data

        Parameters
        ----------
        days : int, optional
            Number of days to simulate (default: 180)
        seed : int, optional
            Random seed for reproducibility (default: 123)
        """
        self.seed = seed
        self._n_days = days
        self.data = None
        self._generate_data()

    def _generate_data(self) -> None:
        """Internal data generation function"""
        np.random.seed(self.seed)
        n = self._n_days

        # Generate timestamps
        start_date = datetime(2025, 1, 1)
        timestamps = [start_date + timedelta(days=i) for i in range(n)]

        # Generate numeric variables
        hr_mean = np.round(np.random.normal(70, 10, n), 1)
        steps = np.round(np.random.normal(7000, 3000, n)).astype(int)
        skin_temp = np.round(np.random.normal(36.5, 0.4, n), 1)
        ambient_temp = np.round(np.random.normal(23, 3, n), 1)
        air_quality_index = np.round(np.random.uniform(20, 120, n)).astype(int)

        # Generate activity labels based on steps
        activity_label = pd.cut(
            steps,
            bins=[-np.inf, 3000, 7000, np.inf],
            labels=["rest", "walk", "exercise"],
        )

        # Generate mood score with dependencies
        mood_score = (
            3
            + 0.001 * (steps - 7000)
            - 0.01 * (air_quality_index - 50)
            + np.random.normal(0, 0.5, n)
        )
        mood_score = np.round(np.clip(mood_score, 1, 5)).astype(int)

        # Generate mood notes
        mood_phrases = [
            "Felt great today.",
            "Very tired.",
            "Worked out hard.",
            "Anxious and stressed.",
            "Calm and productive day.",
            "Slept poorly.",
            "Long day at work.",
        ]

        mood_note = []
        for ms in mood_score:
            if ms >= 4:
                mood_note.append(
                    np.random.choice(
                        [mood_phrases[0], mood_phrases[2], mood_phrases[4]]
                    )
                )
            elif ms <= 2:
                mood_note.append(
                    np.random.choice(
                        [mood_phrases[1], mood_phrases[3], mood_phrases[5]]
                    )
                )
            else:
                mood_note.append(np.random.choice(mood_phrases))

        # Generate sleep quality (dependent variable)
        activity_penalty = np.where(activity_label == "exercise", -10, 0)

        sleep_quality = (
            100
            - 0.2 * hr_mean
            + 0.01 * steps
            + 5 * (mood_score - 3)
            - 0.3 * air_quality_index
            + activity_penalty
            + np.random.normal(0, 5, n)
        )
        sleep_quality = np.round(np.clip(sleep_quality, 0, 100), 1)

        # Create DataFrame
        self.data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "hr_mean": hr_mean,
                "steps": steps,
                "skin_temp": skin_temp,
                "ambient_temp": ambient_temp,
                "activity_label": activity_label,
                "mood_score": mood_score,
                "mood_note": mood_note,
                "air_quality_index": air_quality_index,
                "sleep_quality": sleep_quality,
            }
        )

    def plot_time_series(self, vars: Optional[List[str]] = None) -> plt.Figure:
        """
        Plot time series of selected numeric variables

        Parameters
        ----------
        vars : list of str, optional
            Column names to plot (default: ["hr_mean", "steps", "sleep_quality"])

        Returns
        -------
        matplotlib.figure.Figure
            The time series plot
        """
        if vars is None:
            vars = ["hr_mean", "steps", "sleep_quality"]

        fig, axes = plt.subplots(len(vars), 1, figsize=(12, 3 * len(vars)))
        if len(vars) == 1:
            axes = [axes]

        for i, var in enumerate(vars):
            axes[i].plot(self.data["timestamp"], self.data[var], linewidth=2)
            axes[i].set_title(f"Time Series of {var}")
            axes[i].set_xlabel("Date")
            axes[i].set_ylabel(var)
            axes[i].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig

    def plot_mood_sleep(self) -> plt.Figure:
        """
        Plot the relationship between mood score and sleep quality

        Returns
        -------
        matplotlib.figure.Figure
            Scatter plot with regression line
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        # Create jitter for discrete mood scores
        mood_jitter = self.data["mood_score"] + np.random.normal(
            0, 0.1, len(self.data)
        )

        sns.regplot(
            x=mood_jitter,
            y=self.data["sleep_quality"],
            scatter_kws={"alpha": 0.6, "color": "steelblue"},
            line_kws={"color": "darkred"},
            ax=ax,
        )

        ax.set_xlabel("Mood Score")
        ax.set_ylabel("Sleep Quality")
        ax.set_title("Sleep Quality vs. Mood Score")
        ax.set_xticks(range(1, 6))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_activity_distribution(self) -> plt.Figure:
        """
        Plot the distribution of activity labels

        Returns
        -------
        matplotlib.figure.Figure
            Bar chart of activity distribution
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        activity_counts = self.data["activity_label"].value_counts()
        colors = plt.cm.Set2(np.linspace(0, 1, len(activity_counts)))

        bars = ax.bar(
            activity_counts.index, activity_counts.values, color=colors
        )
        ax.set_xlabel("Activity")
        ax.set_ylabel("Count")
        ax.set_title("Activity Label Distribution")

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        return fig

    def plot_mood_wordcloud(self) -> plt.Figure:
        """
        Create a word cloud from the self-reported mood notes

        Returns
        -------
        matplotlib.figure.Figure
            Word cloud visualization
        """
        try:
            # Combine all mood notes
            text = " ".join(self.data["mood_note"].tolist())

            # Generate word cloud
            wordcloud = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="viridis",
                max_words=100,
            ).generate(text)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.set_title("Mood Notes Word Cloud", fontsize=16)
            ax.axis("off")

            plt.tight_layout()
            return fig

        except ImportError:
            warnings.warn(
                "wordcloud package not installed. Install with: pip install wordcloud"
            )
            return None

    def __repr__(self) -> str:
        return f"SmartHealthSimulator(days={self._n_days}, seed={self.seed})"

    def __str__(self) -> str:
        return f"SmartHealthSimulator with {len(self.data)} days of synthetic health data"
