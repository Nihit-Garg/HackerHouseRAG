export default function IconCircle({ icon: Icon, size = 40, iconSize = 18, className = "" }) {
  return (
    <div
      className={`icon-circle ${className}`}
      style={{ width: size, height: size, borderRadius: size * 0.32 }}
    >
      <Icon size={iconSize} strokeWidth={2.2} />
    </div>
  )
}
