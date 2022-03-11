# cd /home/ubuntu/AutoTrainingPipeline;                                                                       
cd "$(dirname "$0")";
printf "\nCurrent Woking Dir: ";
pwd;
printf "\nAdding, Commiting and Pushing the repository to Github \n\n";
git add .;
git commit -m "... automated commit ...";
git push
# https://stackoverflow.com/questions/35942754/how-to-save-username-and-password-in-git-gitextension
### finally update the evaluation submodule
# git submodule update --recursive --remote
