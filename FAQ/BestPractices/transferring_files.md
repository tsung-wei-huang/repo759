# Transferring Files to/from Euler

There are many methods which you can use to move data on and off of Euler. For the sake of simplicity, these instructions will detail how to do it using the [Recommended Software](software.md#recommended-client-software) tools for this course.


## OpenSSH Tools

OpenSSH -- the default SSH implementation on Windows, Mac, and Linux -- provides a set of common tools for securely moving files over an SSH connection.

### `scp`
Secure Copy, (or `scp`) is used to transfer files between remote and/or local systems by extending the syntax of the familiar `cp` shell command.

#### **Basic Syntax**

Consider the following commands:
```sh
cp original.txt copy.txt
```
versus
```sh
scp original.txt dan@euler.engr.wisc.edu:copy.txt
```

Both examples create a copy of the file `original.txt`. The second example, however, uses a bit of syntactic sugar to specify the computer on which the copy should be created. In this case, it uses the same `user@server` specifier used in the `ssh` command, followed by a colon `:` character. In the second example, `scp` copies the file `original.txt` to the user `dan`'s home directory on `euler.engr.wisc.edu` and calls that file `copy.txt`.


#### **Complex Paths**

Just like the `cp` command, `scp` is capable of handling more complex paths than just the file's base name.

```sh
scp task*.cpp dan@euler.engr.wisc.edu:~/me759-dan/HW02/
```
This example uses the `*` character to find each file that matches the pattern **`task` _something_ `.cpp`**. It also specifies a path into a destination folder (`~/me759-dan/HW02`) on Euler, rather than just leaving the file in the user's home directory.

Note: when the destination file isn't specified, `scp` behaves the same as `cp`... it just reuses the original file's name at the destination.

#### **Reversing Direction**

`scp` isn't limited exclusively to moving files onto a remote server. It can also fetch files from the remote and move them back.

```sh
scp dan@euler.engr.wisc.edu:original.txt copy.txt
```

This example copies the file from Euler to the local directory. All that changed is that the `user@server:` specifier was moved to the source file instead of the destination.

#### **Moving Entire Directories**

Once again, `scp` behaves just like `cp` when copying entire folders.

```sh
scp -r originals/ dan@euler.engr.wisc.edu:copies
```

This example copies the entirety of a folder called `originals` to a directory on euler called `copies`. Both the folder itself and its contents are copied, so this command is useful for moving data in bulk.



