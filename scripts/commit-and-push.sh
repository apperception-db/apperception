diff=0

if ! git diff --exit-code site src test test-runtime; then
  git add --all
  git config --global user.name 'Github Actions Bot'
  git config --global user.email 'apperception-actions-bot@users.noreply.github.com'
  git commit -m "style: $1"
  diff=1
fi

git status

git pull --rebase
git push

exit diff