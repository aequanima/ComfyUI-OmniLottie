import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Load lottie-web from CDN for the Visualizer
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

// --- UX/UI Custom Theme Definition ---
const OMNI_THEME = {
    bgcolor: "#eee8d5",        // Soft off-white base
    title_color: "#1874bb",    // Primary blue
    text_color: "#4f6066",     // Slate text
    accent_color: "#b64a16",   // Orange action color
    success: "#6b8700",
    error: "#b00020"
};

// --- Helper: Dynamic Widget Hiding ---
function toggleWidget(node, widgetName, show) {
    const widget = node.widgets?.find(w => w.name === widgetName);
    if (!widget) return;
    
    // ComfyUI hack to hide/show widgets
    if (show) {
        widget.type = widget.origType || widget.type;
        widget.computeSize = () => [200, 20];
    } else {
        if (widget.type !== "hidden") widget.origType = widget.type;
        widget.type = "hidden";
        widget.computeSize = () => [0, -4]; // Hide spacing
    }
    // Force node resize
    node.size[1] = node.computeSize()[1];
    node.setDirtyCanvas(true, true);
}

app.registerExtension({
    name: "OmniLottie.UX",
    
    // 1. Theme Injection for all OmniLottie nodes
    async nodeCreated(node) {
        if (node.comfyClass && node.comfyClass.startsWith("OmniLottie")) {
            node.color = OMNI_THEME.bgcolor;
            node.bgcolor = OMNI_THEME.title_color;
            node.title_text_color = "#ffffff";
        }
    },

    // 2. Dynamic Widget Logic & Visualizer
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        
        // --- OmniLottie Editor (Dynamic Color Fields) ---
        if (nodeData.name === "OmniLottieEditor") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Initial hide
                setTimeout(() => {
                    toggleWidget(this, "old_hex_A", false);
                    toggleWidget(this, "new_hex_A", false);
                    toggleWidget(this, "old_hex_B", false);
                    toggleWidget(this, "new_hex_B", false);
                }, 10);
                
                return r;
            };

            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (name, value) {
                if (name === "color_swap_A") {
                    toggleWidget(this, "old_hex_A", value);
                    toggleWidget(this, "new_hex_A", value);
                }
                if (name === "color_swap_B") {
                    toggleWidget(this, "old_hex_B", value);
                    toggleWidget(this, "new_hex_B", value);
                }
                if (onWidgetChanged) onWidgetChanged.apply(this, arguments);
            };
        }

        // --- OmniLottie Utility Hub (Dynamic Filename + Heatmap) ---
        if (nodeData.name === "OmniLottieUtilityHub") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Hide filename by default
                setTimeout(() => toggleWidget(this, "filename", false), 10);
                
                // Create VRAM Heatmap Widget
                const heatmapContainer = document.createElement("div");
                heatmapContainer.style.width = "100%";
                heatmapContainer.style.height = "12px";
                heatmapContainer.style.backgroundColor = OMNI_THEME.text_color;
                heatmapContainer.style.borderRadius = "4px";
                heatmapContainer.style.overflow = "hidden";
                heatmapContainer.style.marginTop = "5px";
                
                const bar = document.createElement("div");
                bar.style.width = "0%";
                bar.style.height = "100%";
                bar.style.backgroundColor = OMNI_THEME.success;
                bar.style.transition = "width 0.3s ease, background-color 0.3s ease";
                heatmapContainer.appendChild(bar);

                this.addDOMWidget("vram_heatmap", "heatmap", heatmapContainer);
                
                // Hook execution to update heatmap
                const onExecuted = this.onExecuted;
                this.onExecuted = function (message) {
                    if (onExecuted) onExecuted.apply(this, arguments);
                    if (message?.vram_usage) {
                        const percent = message.vram_usage[0];
                        bar.style.width = `${Math.min(100, Math.max(0, percent))}%`;
                        
                        // Color coding based on A770 limits
                        if (percent < 70) bar.style.backgroundColor = OMNI_THEME.success;
                        else if (percent < 90) bar.style.backgroundColor = OMNI_THEME.accent_orange;
                        else bar.style.backgroundColor = OMNI_THEME.error;
                    }
                };
                
                return r;
            };

            const onWidgetChanged = nodeType.prototype.onWidgetChanged;
            nodeType.prototype.onWidgetChanged = function (name, value) {
                if (name === "mode") {
                    toggleWidget(this, "filename", value === "Load From Input");
                }
                if (onWidgetChanged) onWidgetChanged.apply(this, arguments);
            };
        }

        // --- OmniLottie Visualizer (Player + Quick Export) ---
        if (nodeData.name === "OmniLottieVisualizer") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                
                // Create container
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.height = "300px";
                container.style.backgroundColor = "#ffffff"; // White looks best for most vector art
                container.style.border = `2px solid ${OMNI_THEME.title_color}`;
                container.style.borderRadius = "8px";
                container.style.overflow = "hidden";
                container.style.marginTop = "10px";
                container.style.display = "flex";
                container.style.alignItems = "center";
                container.style.justifyContent = "center";
                container.style.position = "relative";
                
                // Control Overlay UI
                const controls = document.createElement("div");
                controls.style.position = "absolute";
                controls.style.top = "5px";
                controls.style.right = "5px";
                controls.style.display = "flex";
                controls.style.gap = "5px";
                
                let currentJsonData = null;

                const btnStyle = `
                    background: ${OMNI_THEME.bgcolor}; 
                    color: ${OMNI_THEME.title_color}; 
                    border: 1px solid ${OMNI_THEME.title_color}; 
                    border-radius: 4px; 
                    padding: 2px 6px; 
                    font-size: 10px; 
                    cursor: pointer;
                    font-weight: bold;
                `;

                const btnCopy = document.createElement("button");
                btnCopy.innerText = "📋 Copy";
                btnCopy.style.cssText = btnStyle;
                btnCopy.onclick = (e) => {
                    e.stopPropagation();
                    if (currentJsonData) {
                        navigator.clipboard.writeText(currentJsonData);
                        btnCopy.innerText = "✅ Copied";
                        setTimeout(() => btnCopy.innerText = "📋 Copy", 2000);
                    }
                };

                const btnSave = document.createElement("button");
                btnSave.innerText = "⬇️ Save";
                btnSave.style.cssText = btnStyle;
                btnSave.onclick = (e) => {
                    e.stopPropagation();
                    if (currentJsonData) {
                        const blob = new Blob([currentJsonData], { type: "application/json" });
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement("a");
                        a.href = url;
                        a.download = `OmniLottie_QuickExport_${Date.now()}.json`;
                        a.click();
                        URL.revokeObjectURL(url);
                    }
                };

                controls.appendChild(btnCopy);
                controls.appendChild(btnSave);
                container.appendChild(controls);

                this.addDOMWidget("lottie_preview", "preview", container);
                
                let animation = null;

                const renderLottie = async (jsonStr) => {
                    await loadScript(LOTTIE_SCRIPT_URL);
                    currentJsonData = jsonStr;
                    
                    if (animation) animation.destroy();

                    try {
                        const animationData = JSON.parse(jsonStr);
                        animation = window.lottie.loadAnimation({
                            container: container,
                            renderer: 'svg',
                            loop: true,
                            autoplay: true,
                            animationData: animationData
                        });
                        
                        // Ensure controls stay on top
                        container.appendChild(controls);
                        
                        container.onclick = () => {
                            if (animation.isPaused) animation.play();
                            else animation.pause();
                        };
                        
                    } catch (e) {
                        console.error("OmniLottie: Invalid JSON", e);
                        const err = document.createElement("div");
                        err.style.color = OMNI_THEME.error;
                        err.innerText = "Render Error";
                        container.appendChild(err);
                    }
                };

                const onExecuted = this.onExecuted;
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
