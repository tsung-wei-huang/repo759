
### Turning in Homework

All source code files should be located in the appropriate `HWXX` directory with no subdirectories. We will grade the latest commit on the `main` branch as cloned at the due time of the assignment. If you have any doubt about what will be received for grading, you can clone a new copy of your repo and you will see exactly what we will see for grading.

---

### Git Basics

Stage `file` for commit:
```
git add file
```

Commit all staged files:
```
git commit -m "my message"
```

Update the remote (GitLab instance) with all of your local commits:
```
git push
```

Update your local repository with all commits on the remote (GitLab instance):
```
git pull
```

If you wish to make sure you see what you want us to see after you submit, you can check your homework submission by cloning and inspecting it
```
git clone your_hw_repo_url hw_check
cd hw_check
```
Then, look through the contents of `hw_check` this is exactly what we will see for grading, no more, no less. Any *code* files asked for in the homework must be in here and correctly named with the correct file structure according to the assignment header. Canvas submissions of code are not a backup and will be ignored. Things like plots and written answers go to Canvas.


#### Typical Workflow
1. Edit and add files until you reach a good milestone
1. `git add` modified and new files
1. `git commit` to save the milestone
1. `git push` to update the remote
1. Repeat until you're done with the homework tasks
1. Check your homework submission
