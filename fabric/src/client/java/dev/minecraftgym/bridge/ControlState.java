package dev.minecraftgym.bridge;

import com.google.gson.JsonObject;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.util.Mth;

record ControlState(
        float moveX,
        float moveZ,
        boolean jump,
        boolean sprint,
        boolean sneak,
        boolean attack,
        boolean use,
        boolean inventory,
        boolean drop,
        float yawDelta,
        float pitchDelta,
        int hotbarSlot,
        float cursorXDelta,
        float cursorYDelta,
        boolean primaryClick,
        boolean secondaryClick) {

    static final ControlState NOOP = new ControlState(
            0, 0, false, false, false, false, false, false, false,
            0, 0, -1, 0, 0, false, false);

    static ControlState fromJson(JsonObject value) {
        return new ControlState(
                number(value, "move_x", 0),
                number(value, "move_z", 0),
                bool(value, "jump"),
                bool(value, "sprint"),
                bool(value, "sneak"),
                bool(value, "attack"),
                bool(value, "use"),
                bool(value, "inventory"),
                bool(value, "drop"),
                number(value, "yaw_delta", 0),
                number(value, "pitch_delta", 0),
                integer(value, "hotbar_slot", -1),
                number(value, "cursor_x_delta", 0),
                number(value, "cursor_y_delta", 0),
                bool(value, "primary_click"),
                bool(value, "secondary_click"));
    }

    static ControlState captureHuman(
            Minecraft minecraft,
            float priorYaw,
            float priorPitch,
            double priorCursorX,
            double priorCursorY) {
        if (minecraft.player == null) {
            return NOOP;
        }
        float moveX = (minecraft.options.keyRight.isDown() ? 1 : 0)
                - (minecraft.options.keyLeft.isDown() ? 1 : 0);
        float moveZ = (minecraft.options.keyUp.isDown() ? 1 : 0)
                - (minecraft.options.keyDown.isDown() ? 1 : 0);
        int hotbar = -1;
        for (int index = 0; index < minecraft.options.keyHotbarSlots.length; index++) {
            if (minecraft.options.keyHotbarSlots[index].isDown()) {
                hotbar = index;
                break;
            }
        }
        float yawDelta = Mth.wrapDegrees(minecraft.player.getYRot() - priorYaw);
        float pitchDelta = minecraft.player.getXRot() - priorPitch;
        float cursorXDelta = 0;
        float cursorYDelta = 0;
        if (minecraft.screen != null) {
            cursorXDelta = (float) ((minecraft.mouseHandler.xpos() - priorCursorX)
                    / Math.max(1, minecraft.getWindow().getScreenWidth()));
            cursorYDelta = (float) ((minecraft.mouseHandler.ypos() - priorCursorY)
                    / Math.max(1, minecraft.getWindow().getScreenHeight()));
        }
        return new ControlState(
                moveX,
                moveZ,
                minecraft.options.keyJump.isDown(),
                minecraft.options.keySprint.isDown(),
                minecraft.options.keyShift.isDown(),
                minecraft.options.keyAttack.isDown(),
                minecraft.options.keyUse.isDown(),
                minecraft.options.keyInventory.isDown(),
                minecraft.options.keyDrop.isDown(),
                yawDelta,
                pitchDelta,
                hotbar,
                cursorXDelta,
                cursorYDelta,
                minecraft.mouseHandler.isLeftPressed(),
                minecraft.mouseHandler.isRightPressed());
    }

    boolean isHumanIntervention() {
        return Math.abs(moveX) > 0.01
                || Math.abs(moveZ) > 0.01
                || jump || sprint || sneak || attack || use || inventory || drop
                || Math.abs(yawDelta) > 0.01 || Math.abs(pitchDelta) > 0.01
                || hotbarSlot >= 0 || primaryClick || secondaryClick;
    }

    void apply(Minecraft minecraft, boolean firstTick, CursorState cursor) {
        if (minecraft.player == null) {
            return;
        }
        minecraft.options.keyLeft.setDown(moveX < -0.25);
        minecraft.options.keyRight.setDown(moveX > 0.25);
        minecraft.options.keyUp.setDown(moveZ > 0.25);
        minecraft.options.keyDown.setDown(moveZ < -0.25);
        minecraft.options.keyJump.setDown(jump);
        minecraft.options.keySprint.setDown(sprint);
        minecraft.options.keyShift.setDown(sneak);
        minecraft.options.keyAttack.setDown(attack);
        minecraft.options.keyUse.setDown(use);

        if (!firstTick) {
            return;
        }
        minecraft.player.setYRot(Mth.wrapDegrees(minecraft.player.getYRot() + yawDelta));
        minecraft.player.setXRot(Mth.clamp(minecraft.player.getXRot() + pitchDelta, -90, 90));
        if (hotbarSlot >= 0 && hotbarSlot < 9) {
            minecraft.player.getInventory().selected = hotbarSlot;
        }
        if (drop) {
            minecraft.player.drop(false);
        }
        if (inventory) {
            if (minecraft.screen instanceof InventoryScreen) {
                minecraft.setScreen(null);
            } else if (minecraft.screen == null) {
                minecraft.setScreen(new InventoryScreen(minecraft.player));
            }
        }
        if (minecraft.screen != null) {
            cursor.move(cursorXDelta, cursorYDelta, minecraft.screen.width, minecraft.screen.height);
            minecraft.screen.mouseMoved(cursor.x(), cursor.y());
            if (primaryClick) {
                minecraft.screen.mouseClicked(cursor.x(), cursor.y(), 0);
            }
            if (secondaryClick) {
                minecraft.screen.mouseClicked(cursor.x(), cursor.y(), 1);
            }
        }
    }

    JsonObject toJson() {
        JsonObject value = new JsonObject();
        value.addProperty("move_x", moveX);
        value.addProperty("move_z", moveZ);
        value.addProperty("jump", jump);
        value.addProperty("sprint", sprint);
        value.addProperty("sneak", sneak);
        value.addProperty("attack", attack);
        value.addProperty("use", use);
        value.addProperty("inventory", inventory);
        value.addProperty("drop", drop);
        value.addProperty("yaw_delta", yawDelta);
        value.addProperty("pitch_delta", pitchDelta);
        value.addProperty("hotbar_slot", hotbarSlot);
        value.addProperty("cursor_x_delta", cursorXDelta);
        value.addProperty("cursor_y_delta", cursorYDelta);
        value.addProperty("primary_click", primaryClick);
        value.addProperty("secondary_click", secondaryClick);
        return value;
    }

    static void releaseInjectedKeys(Minecraft minecraft) {
        minecraft.options.keyLeft.setDown(false);
        minecraft.options.keyRight.setDown(false);
        minecraft.options.keyUp.setDown(false);
        minecraft.options.keyDown.setDown(false);
        minecraft.options.keyJump.setDown(false);
        minecraft.options.keySprint.setDown(false);
        minecraft.options.keyShift.setDown(false);
        minecraft.options.keyAttack.setDown(false);
        minecraft.options.keyUse.setDown(false);
    }

    private static float number(JsonObject object, String key, float fallback) {
        return object.has(key) ? object.get(key).getAsFloat() : fallback;
    }

    private static int integer(JsonObject object, String key, int fallback) {
        return object.has(key) ? object.get(key).getAsInt() : fallback;
    }

    private static boolean bool(JsonObject object, String key) {
        return object.has(key) && object.get(key).getAsBoolean();
    }

    static final class CursorState {
        private double x;
        private double y;
        private boolean initialized;

        void move(float deltaX, float deltaY, int width, int height) {
            if (!initialized) {
                x = width / 2.0;
                y = height / 2.0;
                initialized = true;
            }
            x = Mth.clamp(x + deltaX * width, 0, Math.max(0, width - 1));
            y = Mth.clamp(y + deltaY * height, 0, Math.max(0, height - 1));
        }

        double x() {
            return x;
        }

        double y() {
            return y;
        }

        void reset() {
            initialized = false;
        }
    }
}
