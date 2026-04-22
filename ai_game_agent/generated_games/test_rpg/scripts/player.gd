extends CharacterBody2D
class_name Player

@export var speed: float = 160.0
@export var max_hp: int = 100
@export var max_mp: int = 50

var hp: int = max_hp
var mp: int = max_mp
var gold: int = 0
var facing: Vector2 = Vector2.DOWN

@onready var anim: AnimatedSprite2D = $AnimatedSprite2D

signal stats_changed(hp, mp, gold)
signal interacted(target)

func _physics_process(_delta: float) -> void:
    var dir := Vector2(
        Input.get_axis("move_left", "move_right"),
        Input.get_axis("move_up",   "move_down")
    ).normalized()
    velocity = dir * speed
    if dir != Vector2.ZERO:
        facing = dir
    _update_animation(dir)
    move_and_slide()
    if Input.is_action_just_pressed("interact"):
        _try_interact()

func _update_animation(dir: Vector2) -> void:
    if dir == Vector2.ZERO:
        anim.play("idle_" + _facing_name())
    else:
        anim.play("walk_" + _facing_name())

func _facing_name() -> String:
    if abs(facing.x) > abs(facing.y):
        return "side"
    return "down" if facing.y > 0 else "up"

func _try_interact() -> void:
    var space := get_world_2d().direct_space_state
    var query := PhysicsRayQueryParameters2D.create(
        global_position,
        global_position + facing * 40.0
    )
    var hit := space.intersect_ray(query)
    if hit and hit.collider:
        interacted.emit(hit.collider)

func take_damage(amount: int) -> void:
    hp = max(0, hp - amount)
    stats_changed.emit(hp, mp, gold)

func heal(amount: int) -> void:
    hp = min(max_hp, hp + amount)
    stats_changed.emit(hp, mp, gold)

func add_gold(amount: int) -> void:
    gold += amount
    stats_changed.emit(hp, mp, gold)
