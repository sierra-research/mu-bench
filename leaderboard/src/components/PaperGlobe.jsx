import { useState, useEffect, useRef, useCallback } from "react";
import { geoOrthographic, geoPath, geoGraticule10 } from "d3-geo";
import * as topojson from "topojson-client";
import "./PaperGlobe.css";

const LOCALE_PINS = [
    { code: "ar-AE", lat: 24.5, lng: 54.7, status: "completed", country: "784" },
    { code: "ar-BH", lat: 26.0, lng: 50.6, status: "completed", country: "048" },
    { code: "ar-KW", lat: 29.3, lng: 47.5, status: "completed", country: "414" },
    { code: "ar-QA", lat: 25.3, lng: 51.2, status: "completed", country: "634" },
    { code: "ar-SA", lat: 23.9, lng: 45.1, status: "completed", country: "682" },
    { code: "az-AZ", lat: 40.4, lng: 49.9, status: "completed", country: "031" },
    { code: "bg-BG", lat: 42.7, lng: 25.5, status: "completed", country: "100" },
    { code: "bn-BD", lat: 23.7, lng: 90.4, status: "completed", country: "050" },
    { code: "bn-IN", lat: 22.6, lng: 88.4, status: "completed", country: "356" },
    { code: "ca-ES", lat: 41.4, lng: 2.2, status: "completed", country: "724" },
    { code: "cs-CZ", lat: 50.1, lng: 14.4, status: "completed", country: "203" },
    { code: "da-DK", lat: 55.7, lng: 12.6, status: "completed", country: "208" },
    { code: "de-AT", lat: 48.2, lng: 16.4, status: "completed", country: "040" },
    { code: "de-CH", lat: 46.9, lng: 7.4, status: "completed", country: "756" },
    { code: "de-DE", lat: 52.5, lng: 13.4, status: "completed", country: "276" },
    { code: "de-LU", lat: 49.6, lng: 6.1, status: "ongoing", country: "442" },
    { code: "el-GR", lat: 37.9, lng: 23.7, status: "completed", country: "300" },
    { code: "en-AU", lat: -33.9, lng: 151.2, status: "completed", country: "036" },
    { code: "en-GB", lat: 51.5, lng: -0.1, status: "completed", country: "826" },
    { code: "en-GB-birmingham", lat: 52.5, lng: -1.9, status: "completed", country: "826" },
    { code: "en-GB-irish-english", lat: 53.3, lng: -6.3, status: "completed", country: "826" },
    { code: "en-GB-liverpool", lat: 53.4, lng: -3.0, status: "completed", country: "826" },
    { code: "en-GB-london", lat: 51.5, lng: -0.1, status: "completed", country: "826" },
    { code: "en-GB-manchester", lat: 53.5, lng: -2.2, status: "completed", country: "826" },
    { code: "en-GB-midlands", lat: 52.6, lng: -1.1, status: "completed", country: "826" },
    { code: "en-GB-newcastle", lat: 55.0, lng: -1.6, status: "completed", country: "826" },
    { code: "en-GB-northern-england", lat: 54.0, lng: -1.5, status: "completed", country: "826" },
    { code: "en-GB-northern-irish", lat: 54.6, lng: -5.9, status: "completed", country: "826" },
    { code: "en-GB-scottish", lat: 55.9, lng: -3.2, status: "completed", country: "826" },
    { code: "en-GB-southern-english", lat: 51.1, lng: -1.3, status: "completed", country: "826" },
    { code: "en-GB-wales", lat: 51.5, lng: -3.2, status: "completed", country: "826" },
    { code: "en-GB-yorkshire", lat: 53.8, lng: -1.5, status: "completed", country: "826" },
    { code: "en-NZ", lat: -36.8, lng: 174.8, status: "completed", country: "554" },
    { code: "en-SG", lat: 1.4, lng: 103.8, status: "completed", country: "702" },
    { code: "en-US", lat: 38.9, lng: -77.0, status: "completed", open: true, country: "840" },
    { code: "es-CO", lat: 4.7, lng: -74.1, status: "ongoing", country: "170" },
    { code: "es-ES", lat: 40.4, lng: -3.7, status: "completed", country: "724" },
    { code: "es-MX", lat: 19.4, lng: -99.1, status: "completed", open: true, country: "484" },
    { code: "es-US", lat: 29.4, lng: -98.5, status: "completed", country: "840" },
    { code: "eu-ES", lat: 43.3, lng: -2.0, status: "completed", country: "724" },
    { code: "fi-FI", lat: 60.2, lng: 24.9, status: "completed", country: "246" },
    { code: "fil-PH", lat: 14.6, lng: 121.0, status: "completed", country: "608" },
    { code: "fr-BE", lat: 50.8, lng: 4.4, status: "completed", country: "056" },
    { code: "fr-CA", lat: 45.5, lng: -73.6, status: "completed", country: "124" },
    { code: "fr-CH", lat: 46.5, lng: 6.6, status: "ongoing", country: "756" },
    { code: "fr-FR", lat: 48.9, lng: 2.4, status: "completed", country: "250" },
    { code: "fr-LU", lat: 49.6, lng: 6.1, status: "ongoing", country: "442" },
    { code: "gl-ES", lat: 42.9, lng: -8.5, status: "completed", country: "724" },
    { code: "he-IL", lat: 31.8, lng: 35.2, status: "completed", country: "376" },
    { code: "hi-IN", lat: 28.6, lng: 77.2, status: "completed", country: "356" },
    { code: "hr-HR", lat: 45.8, lng: 16.0, status: "completed", country: "191" },
    { code: "hu-HU", lat: 47.5, lng: 19.0, status: "completed", country: "348" },
    { code: "id-ID", lat: -6.2, lng: 106.8, status: "completed", country: "360" },
    { code: "is-IS", lat: 64.1, lng: -21.9, status: "ongoing", country: "352" },
    { code: "it-CH", lat: 46.0, lng: 8.9, status: "ongoing", country: "756" },
    { code: "it-IT", lat: 41.9, lng: 12.5, status: "completed", country: "380" },
    { code: "ja-JP", lat: 35.7, lng: 139.7, status: "completed", country: "392" },
    { code: "ko-KR", lat: 37.6, lng: 127.0, status: "completed", country: "410" },
    { code: "lt-LT", lat: 54.7, lng: 25.3, status: "ongoing", country: "440" },
    { code: "lv-LV", lat: 56.9, lng: 24.1, status: "completed", country: "428" },
    { code: "ms-MY", lat: 3.1, lng: 101.7, status: "completed", country: "458" },
    { code: "nb-NO", lat: 59.9, lng: 10.8, status: "completed", country: "578" },
    { code: "nl-BE", lat: 50.8, lng: 4.4, status: "ongoing", country: "056" },
    { code: "nl-NL", lat: 52.4, lng: 4.9, status: "completed", country: "528" },
    { code: "pl-PL", lat: 52.2, lng: 21.0, status: "completed", country: "616" },
    { code: "pt-BR", lat: -15.8, lng: -47.9, status: "completed", country: "076" },
    { code: "pt-PT", lat: 38.7, lng: -9.1, status: "completed", country: "620" },
    { code: "ro-RO", lat: 44.4, lng: 26.1, status: "completed", country: "642" },
    { code: "ru-KZ", lat: 51.2, lng: 71.4, status: "completed", country: "398" },
    { code: "ru-RU", lat: 55.8, lng: 37.6, status: "completed", country: "643" },
    { code: "sk-SK", lat: 48.1, lng: 17.1, status: "ongoing", country: "703" },
    { code: "sv-SE", lat: 59.3, lng: 18.1, status: "completed", country: "752" },
    { code: "th-TH", lat: 13.8, lng: 100.5, status: "completed", country: "764" },
    { code: "tl-PH", lat: 14.6, lng: 121.0, status: "completed", country: "608" },
    { code: "tr-TR", lat: 41.0, lng: 28.9, status: "completed", open: true, country: "792" },
    { code: "uk-UA", lat: 50.5, lng: 30.5, status: "completed", country: "804" },
    { code: "vi-VN", lat: 21.0, lng: 105.9, status: "completed", open: true, country: "704" },
    { code: "zh-CN", lat: 39.9, lng: 116.4, status: "completed", open: true, country: "156" },
    { code: "zh-HK", lat: 22.3, lng: 114.2, status: "completed", country: "344" },
];

const STATUS_COLORS = {
    completed: { fill: "rgba(16,185,129,0.55)", glow: "rgba(16,185,129,0.22)", stroke: "rgba(6,95,70,0.7)" },
    ongoing: { fill: "rgba(245,158,11,0.50)", glow: "rgba(245,158,11,0.18)", stroke: "rgba(180,100,0,0.6)" },
};

const STATUS_PRIORITY = { completed: 2, ongoing: 1 };

const WIDTH = 420;
const HEIGHT = 420;
const COUNTRIES_URL = "https://cdn.jsdelivr.net/npm/world-atlas@2/countries-110m.json";

function buildCountryStatusMap() {
    const map = {};
    for (const pin of LOCALE_PINS) {
        const existing = map[pin.country];
        if (!existing || STATUS_PRIORITY[pin.status] > STATUS_PRIORITY[existing.status]) {
            map[pin.country] = { status: pin.status, open: pin.open || existing?.open };
        } else if (existing && pin.open) {
            existing.open = true;
        }
    }
    return map;
}

const COUNTRY_STATUS = buildCountryStatusMap();

export default function PaperGlobe() {
    const canvasRef = useRef(null);
    const [worldData, setWorldData] = useState(null);
    const [tooltip, setTooltip] = useState(null);
    const rotationRef = useRef([0, -20, 0]);
    const draggingRef = useRef(false);
    const lastMouseRef = useRef(null);
    const animFrameRef = useRef(null);
    const autoRotateRef = useRef(true);

    useEffect(() => {
        fetch(COUNTRIES_URL)
            .then((r) => r.json())
            .then((world) => {
                const countries = topojson.feature(world, world.objects.countries);
                const land = topojson.feature(world, world.objects.land);
                setWorldData({ countries, land });
            })
            .catch(() => {});
    }, []);

    const draw = useCallback(() => {
        const canvas = canvasRef.current;
        if (!canvas || !worldData) return;
        const ctx = canvas.getContext("2d");
        const dpr = window.devicePixelRatio || 1;
        canvas.width = WIDTH * dpr;
        canvas.height = HEIGHT * dpr;
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const projection = geoOrthographic()
            .translate([WIDTH / 2, HEIGHT / 2])
            .scale(WIDTH / 2.15)
            .rotate(rotationRef.current)
            .clipAngle(90);
        const path = geoPath(projection, ctx);
        const globeRadius = WIDTH / 2.15;

        ctx.clearRect(0, 0, WIDTH, HEIGHT);

        // Ocean
        ctx.beginPath();
        ctx.arc(WIDTH / 2, HEIGHT / 2, globeRadius, 0, 2 * Math.PI);
        ctx.fillStyle = "#f0f4f8";
        ctx.fill();

        // Graticule
        ctx.beginPath();
        path(geoGraticule10());
        ctx.strokeStyle = "rgba(0,0,0,0.04)";
        ctx.lineWidth = 0.5;
        ctx.stroke();

        // Uncovered land
        ctx.beginPath();
        path(worldData.land);
        ctx.fillStyle = "#dde4ec";
        ctx.fill();
        ctx.strokeStyle = "#c0c8d4";
        ctx.lineWidth = 0.3;
        ctx.stroke();

        // Covered countries (filled by status)
        for (const feature of worldData.countries.features) {
            const info = COUNTRY_STATUS[feature.id];
            if (!info) continue;
            const colors = STATUS_COLORS[info.status];
            ctx.beginPath();
            path(feature);
            ctx.fillStyle = colors.fill;
            ctx.fill();
            ctx.strokeStyle = colors.stroke;
            ctx.lineWidth = 0.6;
            ctx.stroke();
        }

        // Radial glow around each locale pin for visual presence
        for (const pin of LOCALE_PINS) {
            const coords = projection([pin.lng, pin.lat]);
            if (!coords) continue;
            const dist = Math.sqrt((coords[0] - WIDTH / 2) ** 2 + (coords[1] - HEIGHT / 2) ** 2);
            if (dist > globeRadius) continue;

            const colors = STATUS_COLORS[pin.status];
            const glowR = pin.open ? 28 : 20;
            const grad = ctx.createRadialGradient(coords[0], coords[1], 0, coords[0], coords[1], glowR);
            grad.addColorStop(0, pin.open ? "rgba(99,102,241,0.35)" : colors.glow);
            grad.addColorStop(1, "rgba(0,0,0,0)");
            ctx.beginPath();
            ctx.arc(coords[0], coords[1], glowR, 0, 2 * Math.PI);
            ctx.fillStyle = grad;
            ctx.fill();
        }

        // Small dot pins on top
        for (const pin of LOCALE_PINS) {
            const coords = projection([pin.lng, pin.lat]);
            if (!coords) continue;
            const dist = Math.sqrt((coords[0] - WIDTH / 2) ** 2 + (coords[1] - HEIGHT / 2) ** 2);
            if (dist > globeRadius) continue;

            const r = pin.open ? 4 : 3;
            const dotColor = pin.open ? "#4f46e5" : pin.status === "completed" ? "#059669" : "#d97706";

            ctx.beginPath();
            ctx.arc(coords[0], coords[1], r, 0, 2 * Math.PI);
            ctx.fillStyle = dotColor;
            ctx.fill();

            if (pin.open) {
                ctx.strokeStyle = "white";
                ctx.lineWidth = 1.5;
                ctx.stroke();
            }
        }
    }, [worldData]);

    useEffect(() => {
        if (!worldData) return;
        let last = performance.now();
        function animate(now) {
            if (autoRotateRef.current && !draggingRef.current) {
                const dt = now - last;
                rotationRef.current = [
                    rotationRef.current[0] - dt * 0.008,
                    rotationRef.current[1],
                    rotationRef.current[2],
                ];
            }
            last = now;
            draw();
            animFrameRef.current = requestAnimationFrame(animate);
        }
        animFrameRef.current = requestAnimationFrame(animate);
        return () => {
            if (animFrameRef.current) cancelAnimationFrame(animFrameRef.current);
        };
    }, [worldData, draw]);

    function handlePointerDown(e) {
        draggingRef.current = true;
        autoRotateRef.current = false;
        lastMouseRef.current = { x: e.clientX, y: e.clientY };
        e.currentTarget.setPointerCapture(e.pointerId);
    }

    function handlePointerMove(e) {
        const canvas = canvasRef.current;
        if (!canvas || !worldData) return;

        if (draggingRef.current && lastMouseRef.current) {
            const dx = e.clientX - lastMouseRef.current.x;
            const dy = e.clientY - lastMouseRef.current.y;
            const scale = 0.3;
            rotationRef.current = [
                rotationRef.current[0] - dx * scale,
                Math.max(-89, Math.min(89, rotationRef.current[1] + dy * scale)),
                rotationRef.current[2],
            ];
            lastMouseRef.current = { x: e.clientX, y: e.clientY };
        } else {
            const rect = canvas.getBoundingClientRect();
            const scaleX = WIDTH / rect.width;
            const scaleY = HEIGHT / rect.height;
            const mx = (e.clientX - rect.left) * scaleX;
            const my = (e.clientY - rect.top) * scaleY;

            const projection = geoOrthographic()
                .translate([WIDTH / 2, HEIGHT / 2])
                .scale(WIDTH / 2.15)
                .rotate(rotationRef.current)
                .clipAngle(90);

            let closest = null;
            let closestDist = 20;
            for (const pin of LOCALE_PINS) {
                const coords = projection([pin.lng, pin.lat]);
                if (!coords) continue;
                const d = Math.sqrt((coords[0] - mx) ** 2 + (coords[1] - my) ** 2);
                if (d < closestDist) {
                    closestDist = d;
                    closest = pin;
                }
            }

            if (closest) {
                setTooltip({
                    x: e.clientX - rect.left,
                    y: e.clientY - rect.top - 10,
                    pin: closest,
                });
            } else {
                setTooltip(null);
            }
        }
    }

    function handlePointerUp() {
        draggingRef.current = false;
        lastMouseRef.current = null;
        autoRotateRef.current = true;
    }

    const completed = LOCALE_PINS.filter((p) => p.status === "completed").length;
    const ongoing = LOCALE_PINS.filter((p) => p.status === "ongoing").length;

    return (
        <div className="globe-widget">
            <div className="globe-container">
                <canvas
                    ref={canvasRef}
                    className="globe-canvas"
                    style={{ width: WIDTH, height: HEIGHT }}
                    onPointerDown={handlePointerDown}
                    onPointerMove={handlePointerMove}
                    onPointerUp={handlePointerUp}
                    onPointerLeave={() => {
                        handlePointerUp();
                        setTooltip(null);
                    }}
                />
                {tooltip && (
                    <div className="globe-tooltip" style={{ left: tooltip.x, top: tooltip.y }}>
                        <span className="globe-tooltip-code">{tooltip.pin.code}</span>
                        <span
                            className="globe-tooltip-status"
                            style={{
                                color: tooltip.pin.status === "completed" ? "#10b981" : "#f59e0b",
                            }}
                        >
                            {tooltip.pin.status === "completed" ? "Data collected" : "In progress"}
                        </span>
                        {tooltip.pin.open && <span className="globe-tooltip-open">Open-sourced</span>}
                    </div>
                )}
            </div>
            <div className="globe-legend">
                <span className="globe-legend-item">
                    <span className="globe-legend-dot" style={{ background: "#10b981" }} />
                    Data collected ({completed})
                </span>
                <span className="globe-legend-item">
                    <span className="globe-legend-dot" style={{ background: "#f59e0b" }} />
                    In progress ({ongoing})
                </span>
                <span className="globe-legend-item">
                    <span className="globe-legend-dot globe-legend-dot--open" />
                    Open-sourced (5)
                </span>
            </div>
            <p className="globe-caption">
                {LOCALE_PINS.length} locale variants across 42 languages, collected from Sierra's multilingual voice
                agent. Drag to rotate.
            </p>
        </div>
    );
}
