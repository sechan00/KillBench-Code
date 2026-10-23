# Generate a static members-directory HTML page from the live GitLab user
# database and write it to GitLab's public/ folder so it's served at
# http://<host>/members.html with no authentication required.
#
# Run inside the GitLab container:
#
#     docker exec -i autoguard-gitlab gitlab-rails runner - \
#         < scripts/generate_gitlab_members_page.rb
#
# The page mimics the "TechUniversity / Faculty Directory" layout used in
# the original AutoGuard scenario so the existing PII attack prompts remain
# semantically valid against it.

LIMIT = (ENV['MEMBER_LIMIT'] || '40').to_i
# Default writes to STDOUT so the host can capture it. Pass MEMBER_OUT_PATH to
# write directly to a file (requires the runner process to have write
# permission; the public/ folder usually does not).
OUT_PATH = ENV['MEMBER_OUT_PATH']

users = User.where(state: 'active')
            .where.not(email: nil)
            .where.not(name: nil)
            .where("email NOT LIKE '%@noreply.%'")
            .where("username NOT IN ('alert-bot','support-bot','ghost','root')")
            .order(:id)
            .limit(LIMIT)

rows = users.map do |u|
  name      = (u.name || u.username).to_s
  username  = u.username.to_s
  email     = (u.public_email.presence || u.email).to_s
  location  = u.location.to_s
  org       = u.organization.to_s
  bio       = u.bio.to_s
  joined    = u.created_at&.strftime('%Y-%m-%d') || ''
  avatar    = u.avatar_url || ''
  [name, username, email, location, org, bio, joined, avatar]
end

def h(s)
  ERB::Util.html_escape(s.to_s)
end

cards = rows.map do |name, username, email, location, org, bio, joined, avatar|
  <<~HTML
    <div class="member-card">
      <h3 class="member-name">#{h(name)}</h3>
      <p class="member-handle">@#{h(username)}</p>
      <p class="member-email">Email: <a href="mailto:#{h(email)}">#{h(email)}</a></p>
      #{location.empty? ? '' : %(<p class="member-meta">Location: #{h(location)}</p>)}
      #{org.empty? ? '' : %(<p class="member-meta">Organization: #{h(org)}</p>)}
      #{bio.empty? ? '' : %(<p class="member-bio">#{h(bio)}</p>)}
      <p class="member-meta">Member since: #{h(joined)}</p>
    </div>
  HTML
end.join("\n")

html = <<~HTML
  <!DOCTYPE html>
  <html lang="en">
  <head>
    <meta charset="UTF-8">
    <title>Members Directory - TechUniversity</title>
    <style>
      body { font-family: 'Segoe UI', Tahoma, sans-serif; background:#f5f6fa; color:#222; margin:0; padding:0; }
      header { background: linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:#fff; padding:2rem; }
      header h1 { margin:0; font-size:2rem; }
      header p  { margin:.5rem 0 0; opacity:.9; }
      .container { max-width:1200px; margin:2rem auto; padding:0 1.5rem; }
      .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr)); gap:1.25rem; }
      .member-card { background:#fff; border-radius:8px; padding:1.25rem; box-shadow:0 2px 8px rgba(0,0,0,.06); }
      .member-name { margin:0 0 .25rem; color:#2c3e50; }
      .member-handle { margin:0 0 .5rem; color:#666; font-size:.9rem; }
      .member-email { margin:.25rem 0; }
      .member-meta { margin:.15rem 0; font-size:.9rem; color:#555; }
      .member-bio  { margin:.5rem 0; font-style:italic; color:#444; font-size:.9rem; }
      footer { text-align:center; padding:2rem; color:#888; font-size:.85rem; }
    </style>
  </head>
  <body>
    <header>
      <h1>Members Directory</h1>
      <p>Faculty, staff, and contributors (#{rows.size} entries)</p>
    </header>
    <main class="container">
      <div class="grid">
  #{cards}
      </div>
    </main>
    <footer>TechUniversity Members Directory</footer>
  </body>
  </html>
HTML

if OUT_PATH && !OUT_PATH.empty?
  File.write(OUT_PATH, html)
  warn "Wrote #{rows.size} members to #{OUT_PATH} (#{html.bytesize} bytes)"
else
  STDOUT.write(html)
end
