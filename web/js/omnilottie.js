import { app } from "../../scripts/app.js";

// Load lottie-web from a CDN
const LOTTIE_SCRIPT_URL = "https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.12.2/lottie.min.js";

function loadScript(url) {
    return new Promise((resolve, reject) => {
        if (window.lottie) return resolve();
        const script = document.createElement("script");
        script.src = url;
        script.onload = resolve;
        script.onerror = reject;
        document.head.appendChild(script);
    });
}

app.registerExtension({
    name: "OmniLottie.Visualizer",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "OmniLottieVisualizer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Create container
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "300px";
                container.style.backgroundColor = "#111";
                container.style.borderRadius = "8px";
                container.style.overflow = "hidden";
                container.style.marginTop = "10px";
                container.style.display = "flex";
                container.style.alignItems = "center";
                container.style.justifyContent = "center";
                container.style.position = "relative";
                
                // Add play/pause hint
                const hint = document.createElement("div");
                hint.innerText = "Click to Play/Pause";
                hint.style.position = "absolute";
                hint.style.bottom = "5px";
                hint.style.fontSize = "10px";
                hint.style.color = "#666";
                container.appendChild(hint);

                this.addDOMWidget("lottie_preview", "preview", container);
                
                let animation = null;

                const renderLottie = async (jsonStr) => {
                    await loadScript(LOTTIE_SCRIPT_URL);
                    
                    if (animation) {
                        animation.destroy();
                    }

                    try {
                        const animationData = JSON.parse(jsonStr);
                        animation = window.lottie.loadAnimation({
                            container: container,
                            renderer: 'svg',
                            loop: true,
                            autoplay: true,
                            animationData: animationData
                        });
                        
                        container.onclick = () => {
                            if (animation.isPaused) animation.play();
                            else animation.pause();
                        };
                        
                    } catch (e) {
                        console.error("OmniLottie: Invalid Lottie JSON", e);
                        container.innerText = "Error Loading Lottie";
                    }
                };

                // Listen for execution data
                const onExecuted = nodeType.prototype.onExecuted;
                this.onExecuted = function (message) {
                    if (onExecuted) onExecuted.apply(this, arguments);
                    if (message?.lottie_json) {
                        renderLottie(message.lottie_json[0]);
                    }
                };
                
                return r;
            };
        }
    }
});
