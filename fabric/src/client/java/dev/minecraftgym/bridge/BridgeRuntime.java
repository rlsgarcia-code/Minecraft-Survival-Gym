package dev.minecraftgym.bridge;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;

import com.google.gson.JsonArray;
import com.google.gson.JsonNull;
import com.google.gson.JsonObject;

import net.minecraft.client.Minecraft;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.server.level.ServerPlayer;
import net.minecraft.world.entity.Entity.RemovalReason;
import net.minecraft.world.entity.RelativeMovement;
import net.minecraft.world.level.GameType;

final class BridgeRuntime implements AutoCloseable {
    static final int PROTOCOL_VERSION = 1;

    private final ConcurrentLinkedQueue<PendingRequest> requests = new ConcurrentLinkedQueue<>();
    private final ControlState.CursorState cursor = new ControlState.CursorState();
    private ActiveStep activeStep;
    private ActiveReset activeReset;
    private int imageWidth = 128;
    private int imageHeight = 128;
    private long worldSeed;
    private float priorHumanYaw;
    private float priorHumanPitch;
    private double priorCursorX;
    private double priorCursorY;
    private List<Integer> priorInventory = List.of();
    private float priorHealth = 1;
    private ResetAnchor resetAnchor;

    CompletableFuture<JsonObject> submit(JsonObject request) {
        CompletableFuture<JsonObject> future = new CompletableFuture<>();
        requests.add(new PendingRequest(request, future));
        return future;
    }

    void onStartTick(Minecraft minecraft) {
        if (activeStep != null) {
            activeStep.onStartTick(minecraft);
            return;
        }
        if (activeReset != null) {
            return;
        }
        PendingRequest pending = requests.poll();
        if (pending == null) {
            return;
        }
        try {
            String type = pending.request().get("type").getAsString();
            switch (type) {
                case "reset" -> beginReset(minecraft, pending);
                case "step" -> beginStep(minecraft, pending);
                case "close" -> {
                    releaseSession();
                    JsonObject response = new JsonObject();
                    response.addProperty("ok", true);
                    pending.future().complete(response);
                }
                default -> throw new IllegalArgumentException("Unknown request type: " + type);
            }
        } catch (Exception error) {
            pending.future().completeExceptionally(error);
        }
    }

    void onEndTick(Minecraft minecraft) {
        if (activeReset != null) {
            try {
                if (activeReset.serverWork().isDone() && activeReset.ready(minecraft)) {
                    finishReset(minecraft);
                }
            } catch (Exception error) {
                ActiveReset failed = activeReset;
                activeReset = null;
                failed.pending().future().completeExceptionally(error);
            }
            return;
        }
        if (activeStep == null) {
            return;
        }
        activeStep.onEndTick(minecraft);
        if (activeStep.finished()) {
            finishStep(minecraft);
        }
    }

    private void beginReset(Minecraft minecraft, PendingRequest pending) {
        requireWorld(minecraft);
        JsonObject request = pending.request();
        JsonObject image = request.getAsJsonObject("image");
        imageWidth = image.get("width").getAsInt();
        imageHeight = image.get("height").getAsInt();
        if (imageWidth <= 0 || imageHeight <= 0 || imageWidth * imageHeight > 4_194_304) {
            throw new IllegalArgumentException("Invalid requested image dimensions");
        }
        MinecraftServer server = requireIntegratedServer(minecraft);
        if (resetAnchor == null) {
            resetAnchor = ResetAnchor.capture(server, minecraft.player.getUUID());
        }
        CompletableFuture<Void> serverWork = new CompletableFuture<>();
        server.execute(() -> {
            try {
                resetPlayerAndWorld(server, minecraft.player.getUUID());
                worldSeed = server.getWorldData().worldGenOptions().seed();
                server.tickRateManager().setFrozen(true);
                serverWork.complete(null);
            } catch (Exception error) {
                serverWork.completeExceptionally(error);
            }
        });
        activeReset = new ActiveReset(pending, serverWork, 3);
    }

    private void finishReset(Minecraft minecraft) {
        ActiveReset reset = activeReset;
        activeReset = null;
        try {
            reset.serverWork().join();
            requireWorld(minecraft);
            cursor.reset();
            priorHumanYaw = minecraft.player.getYRot();
            priorHumanPitch = minecraft.player.getXRot();
            priorCursorX = minecraft.mouseHandler.xpos();
            priorCursorY = minecraft.mouseHandler.ypos();
            priorInventory = inventoryCounts(minecraft);
            priorHealth = minecraft.player.getHealth();
            JsonObject response = transitionResponse(minecraft, false, false, List.of());
            JsonObject info = response.getAsJsonObject("info");
            info.addProperty("reset_mode", "soft_anchor");
            long requestedSeed = reset.pending().request().get("seed").getAsLong();
            info.addProperty("requested_seed", requestedSeed);
            info.addProperty("requested_seed_matches_world", requestedSeed == worldSeed);
            reset.pending().future().complete(response);
        } catch (Exception error) {
            reset.pending().future().completeExceptionally(error);
        }
    }

    private void beginStep(Minecraft minecraft, PendingRequest pending) {
        requireWorld(minecraft);
        MinecraftServer server = requireIntegratedServer(minecraft);
        JsonObject request = pending.request();
        int frameSkip = request.get("frame_skip").getAsInt();
        if (frameSkip <= 0 || frameSkip > 1000) {
            throw new IllegalArgumentException("frame_skip must be in [1, 1000]");
        }
        String mode = request.get("control_mode").getAsString();
        if (!Set.of("agent", "human", "dagger").contains(mode)) {
            throw new IllegalArgumentException("Unknown control_mode: " + mode);
        }
        ControlState policyAction = ControlState.fromJson(request.getAsJsonObject("action"));
        activeStep = new ActiveStep(pending, server, policyAction, mode, frameSkip);
        activeStep.onStartTick(minecraft);
    }

    private void finishStep(Minecraft minecraft) {
        ActiveStep step = activeStep;
        activeStep = null;
        try {
            ControlState.releaseInjectedKeys(minecraft);
            boolean terminated = minecraft.player == null
                    || minecraft.player.isDeadOrDying()
                    || minecraft.player.getHealth() <= 0;
            List<JsonObject> events = detectEvents(minecraft, terminated);
            JsonObject response = transitionResponse(minecraft, terminated, false, events);
            JsonObject info = response.getAsJsonObject("info");
            info.addProperty("control_source", step.controlSource);
            info.add("policy_action", step.policyAction.toJson());
            if (step.humanAction == null) {
                info.add("human_action", JsonNull.INSTANCE);
            } else {
                info.add("human_action", step.humanAction.toJson());
            }
            info.add("executed_action", step.executedAction.toJson());
            step.pending.future().complete(response);
        } catch (Exception error) {
            step.pending.future().completeExceptionally(error);
        }
    }

    private JsonObject transitionResponse(
            Minecraft minecraft,
            boolean terminated,
            boolean truncated,
            List<JsonObject> events) {
        JsonObject response = new JsonObject();
        response.addProperty("ok", true);
        response.add("observation", ObservationEncoder.capture(
                minecraft, imageWidth, imageHeight, worldSeed));
        response.addProperty("terminated", terminated);
        response.addProperty("truncated", truncated);
        JsonArray eventArray = new JsonArray();
        events.forEach(eventArray::add);
        response.add("events", eventArray);
        response.add("info", new JsonObject());
        return response;
    }

    private List<JsonObject> detectEvents(Minecraft minecraft, boolean died) {
        List<JsonObject> events = new ArrayList<>();
        float currentHealth = minecraft.player == null ? 0 : minecraft.player.getHealth();
        if (currentHealth < priorHealth) {
            JsonObject event = new JsonObject();
            event.addProperty("type", "health_lost");
            event.addProperty("amount", priorHealth - currentHealth);
            events.add(event);
        }
        List<Integer> currentInventory = inventoryCounts(minecraft);
        int slots = Math.min(priorInventory.size(), currentInventory.size());
        for (int slot = 0; slot < slots; slot++) {
            int delta = currentInventory.get(slot) - priorInventory.get(slot);
            if (delta > 0) {
                JsonObject event = new JsonObject();
                event.addProperty("type", "inventory_increased");
                event.addProperty("slot", slot);
                event.addProperty("count", delta);
                events.add(event);
            }
        }
        if (died) {
            JsonObject event = new JsonObject();
            event.addProperty("type", "death");
            events.add(event);
        }
        priorHealth = currentHealth;
        priorInventory = currentInventory;
        return events;
    }

    private static List<Integer> inventoryCounts(Minecraft minecraft) {
        List<Integer> counts = new ArrayList<>(36);
        for (int slot = 0; slot < 36; slot++) {
            counts.add(minecraft.player == null ? 0 : minecraft.player.getInventory().getItem(slot).getCount());
        }
        return counts;
    }

    private void resetPlayerAndWorld(MinecraftServer server, java.util.UUID playerId) {
        ServerPlayer player = server.getPlayerList().getPlayer(playerId);
        if (player == null) {
            throw new IllegalStateException("Server player is unavailable");
        }
        if (player.isDeadOrDying()) {
            player = server.getPlayerList().respawn(player, false, RemovalReason.CHANGED_DIMENSION);
        }
        player.setGameMode(GameType.SURVIVAL);
        player.getInventory().clearContent();
        player.removeAllEffects();
        player.setHealth(player.getMaxHealth());
        player.getFoodData().setFoodLevel(20);
        player.getFoodData().setSaturation(5);
        player.getFoodData().setExhaustion(0);
        player.teleportTo(
                resetAnchor.level(),
                resetAnchor.x(),
                resetAnchor.y(),
                resetAnchor.z(),
                Set.<RelativeMovement>of(),
                resetAnchor.yaw(),
                resetAnchor.pitch());
        ServerLevel overworld = server.overworld();
        overworld.setDayTime(0);
        overworld.setWeatherParameters(6000, 0, false, false);
    }

    void releaseControls() {
        Minecraft minecraft = Minecraft.getInstance();
        minecraft.execute(() -> ControlState.releaseInjectedKeys(minecraft));
    }

    private void releaseSession() {
        Minecraft minecraft = Minecraft.getInstance();
        minecraft.execute(() -> {
            ControlState.releaseInjectedKeys(minecraft);
            MinecraftServer server = minecraft.getSingleplayerServer();
            if (server != null) {
                server.execute(() -> server.tickRateManager().setFrozen(false));
            }
        });
    }

    @Override
    public void close() {
        releaseSession();
        PendingRequest pending;
        while ((pending = requests.poll()) != null) {
            pending.future().completeExceptionally(new IllegalStateException("Bridge closed"));
        }
    }

    private static void requireWorld(Minecraft minecraft) {
        if (minecraft.player == null || minecraft.level == null) {
            throw new IllegalStateException("Open a single-player world before using the bridge");
        }
    }

    private static MinecraftServer requireIntegratedServer(Minecraft minecraft) {
        MinecraftServer server = minecraft.getSingleplayerServer();
        if (server == null) {
            throw new IllegalStateException("Protocol v1 requires an integrated single-player server");
        }
        return server;
    }

    private record PendingRequest(JsonObject request, CompletableFuture<JsonObject> future) {
    }

    private static final class ActiveReset {
        private final PendingRequest pending;
        private final CompletableFuture<Void> serverWork;
        private int settleClientTicks;
        private int readinessTimeout = 200;

        private ActiveReset(
                PendingRequest pending,
                CompletableFuture<Void> serverWork,
                int settleClientTicks) {
            this.pending = pending;
            this.serverWork = serverWork;
            this.settleClientTicks = settleClientTicks;
        }

        PendingRequest pending() {
            return pending;
        }

        CompletableFuture<Void> serverWork() {
            return serverWork;
        }

        boolean ready(Minecraft minecraft) {
            if (settleClientTicks > 0) {
                settleClientTicks--;
                return false;
            }
            if (minecraft.player != null
                    && !minecraft.player.isDeadOrDying()
                    && minecraft.player.getHealth() > 0) {
                return true;
            }
            if (--readinessTimeout <= 0) {
                throw new IllegalStateException("Player did not become ready after reset");
            }
            return false;
        }
    }

    private final class ActiveStep {
        private final PendingRequest pending;
        private final MinecraftServer server;
        private final ControlState policyAction;
        private final String mode;
        private final int frameSkip;
        private ControlState humanAction;
        private ControlState executedAction = ControlState.NOOP;
        private String controlSource = "agent";
        private boolean firstTick = true;
        private boolean actionChosen;
        private boolean serverStepRequested;
        private volatile boolean serverStepStarted;
        private int settleClientTicks = 2;

        private ActiveStep(
                PendingRequest pending,
                MinecraftServer server,
                ControlState policyAction,
                String mode,
                int frameSkip) {
            this.pending = pending;
            this.server = server;
            this.policyAction = policyAction;
            this.mode = mode;
            this.frameSkip = frameSkip;
        }

        void onStartTick(Minecraft minecraft) {
            if (actionChosen) {
                return;
            }
            humanAction = ControlState.captureHuman(
                    minecraft,
                    priorHumanYaw,
                    priorHumanPitch,
                    priorCursorX,
                    priorCursorY);
            priorHumanYaw = minecraft.player.getYRot();
            priorHumanPitch = minecraft.player.getXRot();
            priorCursorX = minecraft.mouseHandler.xpos();
            priorCursorY = minecraft.mouseHandler.ypos();
            if ("human".equals(mode)) {
                executedAction = humanAction;
                controlSource = "human";
            } else if ("dagger".equals(mode) && humanAction.isHumanIntervention()) {
                executedAction = humanAction;
                controlSource = "human";
            } else {
                executedAction = policyAction;
                controlSource = "agent";
                executedAction.apply(minecraft, firstTick, cursor);
            }
            firstTick = false;
            actionChosen = true;
        }

        void onEndTick(Minecraft minecraft) {
            if (!serverStepRequested) {
                serverStepRequested = true;
                server.execute(() -> {
                    server.tickRateManager().setFrozen(true);
                    if (!server.tickRateManager().stepGameIfPaused(frameSkip)) {
                        pending.future().completeExceptionally(
                                new IllegalStateException("Server refused frozen tick step"));
                    } else {
                        serverStepStarted = true;
                    }
                });
                return;
            }
            if (serverStepStarted
                    && server.tickRateManager().frozenTicksToRun() == 0
                    && --settleClientTicks <= 0) {
                ControlState.releaseInjectedKeys(minecraft);
            }
        }

        boolean finished() {
            return serverStepRequested
                    && serverStepStarted
                    && server.tickRateManager().frozenTicksToRun() == 0
                    && settleClientTicks <= 0;
        }
    }

    private record ResetAnchor(
            ServerLevel level,
            double x,
            double y,
            double z,
            float yaw,
            float pitch) {
        static ResetAnchor capture(MinecraftServer server, java.util.UUID playerId) {
            ServerPlayer player = server.getPlayerList().getPlayer(playerId);
            if (player == null) {
                throw new IllegalStateException("Server player is unavailable");
            }
            return new ResetAnchor(
                    player.serverLevel(),
                    player.getX(),
                    player.getY(),
                    player.getZ(),
                    player.getYRot(),
                    player.getXRot());
        }
    }
}
