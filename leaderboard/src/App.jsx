import { useState, useEffect } from "react";
import Leaderboard from "./components/Leaderboard.jsx";
import Paper from "./components/Paper.jsx";
import "./App.css";

function getViewFromHash() {
    return window.location.hash === "#paper" ? "paper" : "leaderboard";
}

export default function App() {
    const [view, setView] = useState(getViewFromHash);

    useEffect(() => {
        const onHashChange = () => setView(getViewFromHash());
        window.addEventListener("hashchange", onHashChange);
        return () => window.removeEventListener("hashchange", onHashChange);
    }, []);

    useEffect(() => {
        window.scrollTo(0, 0);
    }, [view]);

    const navigateTo = (target) => {
        window.location.hash = target === "paper" ? "#paper" : "";
    };

    return (
        <div className="app">
            <nav className="navbar">
                <div className="navbar-inner">
                    <div className="navbar-brand">
                        <span className="navbar-title">
                            <span className="mu-symbol">&mu;</span>-bench
                        </span>
                        <a
                            href="https://sierra.ai"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="navbar-attribution"
                        >
                            <img
                                src={`${import.meta.env.BASE_URL}sierra_logo.jpeg`}
                                alt="Sierra"
                                className="navbar-logo"
                            />
                            <span className="navbar-from">from Sierra</span>
                        </a>
                    </div>
                    <div className="navbar-links">
                        <button
                            onClick={() => navigateTo("leaderboard")}
                            className={`navbar-link${view === "leaderboard" ? " navbar-link-active" : ""}`}
                        >
                            Leaderboard
                        </button>
                        <a
                            href="https://huggingface.co/datasets/sierra-research/mu-bench"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="navbar-link"
                        >
                            Download
                        </a>
                        <a
                            href="https://github.com/sierra-research/mu-bench"
                            target="_blank"
                            rel="noopener noreferrer"
                            className="navbar-link"
                        >
                            Submit
                        </a>
                        <button
                            onClick={() => navigateTo("paper")}
                            className={`navbar-link${view === "paper" ? " navbar-link-active" : ""}`}
                        >
                            Blog
                        </button>
                    </div>
                </div>
            </nav>

            <main className="main-content">
                {view === "leaderboard" ? (
                    <>
                        <div className="home">
                            <section className="hero">
                                <h1 className="hero-main-title">
                                    <span className="mu-symbol">&mu;</span>
                                    <span className="bench-text">-bench</span>
                                </h1>
                                <p className="hero-subtitle">
                                    <span className="hero-highlight">M</span>ultilingual{" "}
                                    <span className="hero-highlight">U</span>tterances Transcription Benchmark
                                </p>
                            </section>
                        </div>
                        <Leaderboard />
                    </>
                ) : (
                    <Paper />
                )}
            </main>

            <footer className="footer">
                <div className="footer-inner">
                    <p className="footer-contact">
                        For questions or feedback, contact{" "}
                        <a href="mailto:soham@sierra.ai" className="footer-email">
                            soham@sierra.ai
                        </a>{" "}
                        or{" "}
                        <a href="https://github.com/sierra-research/mu-bench/issues" className="footer-email">
                            open an issue on GitHub
                        </a>
                    </p>
                </div>
            </footer>
        </div>
    );
}
